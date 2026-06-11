from __future__ import annotations

import queue
import hashlib
import sys
import threading
import time
import traceback
from collections import deque
from copy import deepcopy
from pathlib import Path

import numpy as np
import pyaudio
import streamlit as st
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.offline_app import (
    CKPT_PATH,
    default_target_index,
    find_target_choices,
    packaged_file_from_dir,
    resolve_target_choice,
)
from stylizer.cfm_lightning_module import CFMLightningModule
from stylizer.modules import StreamingCrossfader


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
IO_CHUNK = 240
OVERLAP = 320 * 2
LOOKAHEAD = 80
SPEED_TEST_MIN_STEPS = 6
SPEED_TEST_MAX_STEPS = 16
SPEED_TEST_REPEATS = 3
SPEED_TEST_WARMUP_CHUNKS = 1
SLOW_CHUNK_LIMIT = 5
STEPS_WIDGET_KEY = "streaming_steps"
CFG_WIDGET_KEY = "streaming_cfg_strength"
CHUNK_SIZE_WIDGET_KEY = "streaming_chunk_size"
PREPARED_TARGETS_KEY = "streaming_prepared_target_labels"
ACTIVE_TARGET_WIDGET_KEY = "streaming_active_target_label"


def inject_page_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.25rem;
        }
        div[data-testid="stAudio"] {
            margin-top: 0.15rem;
            margin-bottom: 0.15rem;
        }
        div[data-testid="stButton"] > button {
            min-height: 2.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def mps_is_available() -> bool:
    return (
        sys.platform == "darwin"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and hasattr(torch, "mps")
    )


def default_device() -> str:
    if mps_is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def available_devices() -> list[str]:
    devices = [default_device()]
    for device in ("cuda", "mps", "cpu"):
        if device == "cuda" and not torch.cuda.is_available():
            continue
        if device == "mps" and not mps_is_available():
            continue
        if device not in devices:
            devices.append(device)
    return devices


def synchronize_device(device) -> None:
    device_type = getattr(device, "type", str(device)).split(":", 1)[0]
    if device_type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    elif device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


@st.cache_data(show_spinner=False, ttl=10)
def list_audio_devices(input_device: bool) -> list[dict]:
    pa = pyaudio.PyAudio()
    devices = []
    try:
        for idx in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(idx)
            max_channels = info.get("maxInputChannels", 0) if input_device else info.get("maxOutputChannels", 0)
            if max_channels <= 0:
                continue
            devices.append(
                {
                    "index": idx,
                    "name": info.get("name", "Unknown"),
                    "channels": int(max_channels),
                    "sample_rate": int(float(info.get("defaultSampleRate", RATE))),
                }
            )
    finally:
        pa.terminate()
    return devices


def audio_device_label(device: dict) -> str:
    return f"{device['name']} · {device['channels']} ch · {device['sample_rate']} Hz · #{device['index']}"


@st.cache_resource(show_spinner=False)
def load_streaming_model(checkpoint_path: str, device: str) -> CFMLightningModule:
    model = CFMLightningModule.load_for_inference(checkpoint_path, device=device)
    model.eval()
    return model


def total_needed(chunk_size: int) -> int:
    return chunk_size + OVERLAP + LOOKAHEAD


def model_chunk_frames(chunk_size: int) -> int:
    return max(1, (chunk_size + OVERLAP) // 320)


def max_context_chunks(chunk_size: int) -> int:
    return 150 // model_chunk_frames(chunk_size) + 1


def reset_streaming_state(model: CFMLightningModule, target_path: Path, chunk_size: int) -> None:
    model.clear_cache()
    model.prepare_streaming(str(target_path), chunk_size)


def snapshot_style_state(model: CFMLightningModule) -> dict:
    return {
        "tgt_content": model.tgt_content.clone(),
        "tgt_mel": model.tgt_mel.clone(),
        "ssl_style_emb": model.ssl_style_emb.clone(),
        "destylizer_buffer": deepcopy(model.destylizer_buffer),
    }


def apply_style_state(model: CFMLightningModule, crossfader: StreamingCrossfader, style_state: dict) -> None:
    if hasattr(model, "left_content"):
        del model.left_content
    model.vocoder.clear_cache()
    model.tgt_content = style_state["tgt_content"]
    model.tgt_mel = style_state["tgt_mel"]
    model.ssl_style_emb = style_state["ssl_style_emb"]
    model.destylizer_buffer = deepcopy(style_state["destylizer_buffer"])
    crossfader.reset()


def prepare_style_state(model: CFMLightningModule, target_path: Path, chunk_size: int) -> dict:
    reset_streaming_state(model, target_path, chunk_size)
    return snapshot_style_state(model)


def time_one_realtime_chunk(
    model: CFMLightningModule,
    crossfader: StreamingCrossfader,
    chunk: np.ndarray,
    steps: int,
    cfg_strength: float,
) -> float:
    synchronize_device(model.device)
    start_t = time.perf_counter()
    out_wav = model.realtime_sample(chunk, steps=steps, cfg_strength=cfg_strength)
    crossfader.push(out_wav)
    synchronize_device(model.device)
    return time.perf_counter() - start_t


def run_speed_test(
    model: CFMLightningModule,
    target_path: Path,
    chunk_size: int,
    max_steps: int,
    cfg_strength: float,
    device: str,
    progress_callback=None,
) -> tuple[int, list[tuple[int, float]]]:
    test_chunk = np.random.default_rng(0).normal(0.0, 0.02, total_needed(chunk_size)).clip(-1.0, 1.0).astype(np.float32)
    budget_seconds = chunk_size / RATE
    results = []
    max_steps = max(int(max_steps), SPEED_TEST_MIN_STEPS)
    step_values = list(range(max_steps, SPEED_TEST_MIN_STEPS - 1, -1))

    for idx, steps in enumerate(step_values, start=1):
        reset_streaming_state(model, target_path, chunk_size)
        crossfader = StreamingCrossfader(overlap=OVERLAP, device=device)

        for _ in range(max_context_chunks(chunk_size)):
            time_one_realtime_chunk(model, crossfader, test_chunk, SPEED_TEST_MIN_STEPS, cfg_strength)

        durations = []
        for repeat_idx in range(SPEED_TEST_WARMUP_CHUNKS + SPEED_TEST_REPEATS):
            elapsed = time_one_realtime_chunk(model, crossfader, test_chunk, steps, cfg_strength)
            if repeat_idx >= SPEED_TEST_WARMUP_CHUNKS:
                durations.append(elapsed)
        mean_seconds = float(np.mean(durations))
        results.append((steps, mean_seconds))
        if progress_callback is not None:
            progress_callback(idx, len(step_values), steps, mean_seconds)

    valid_steps = [steps for steps, mean_seconds in results if mean_seconds < budget_seconds]
    reset_streaming_state(model, target_path, chunk_size)
    if not valid_steps:
        raise RuntimeError(
            "This machine cannot support streaming inference: even 6 inference steps "
            f"exceeded the realtime budget of {budget_seconds * 1000:.0f} ms per chunk. "
            "It is simply not fast enough for this streaming setup."
        )
    recommended_steps = max(valid_steps)
    return recommended_steps, results


def streamability_failure_message(elapsed: float, chunk_duration: float, slow_chunks: int) -> str:
    return (
        "Current setup is not streamable: processing is slower than realtime. "
        f"The latest chunk took {elapsed * 1000:.0f} ms, but the chunk duration is "
        f"{chunk_duration * 1000:.0f} ms; this happened for {slow_chunks} consecutive chunks. "
        "Lower inference steps, increase chunk size, choose a faster runtime device, "
        "or enable the speed test before streaming."
    )


class StreamingController:
    def __init__(
        self,
        *,
        target_label: str,
        target_paths: dict[str, Path],
        input_device_index: int,
        output_device_index: int,
        device: str,
        chunk_size: int,
        steps: int,
        cfg_strength: float,
        run_benchmark: bool,
    ):
        self.target_label = target_label
        self.target_paths = dict(target_paths)
        self.input_device_index = input_device_index
        self.output_device_index = output_device_index
        self.device = device
        self.chunk_size = int(chunk_size)
        self.steps = max(SPEED_TEST_MIN_STEPS, int(steps))
        self.cfg_strength = float(cfg_strength)
        self.run_benchmark = bool(run_benchmark)
        self.runtime_steps = self.steps
        self.autotuned_steps: int | None = None
        self.autotuned_version = 0
        self.status = "Starting"
        self.error: str | None = None
        self.slow_chunks = 0
        self.slow_monitor_grace_chunks = 0
        self.processed_chunks = 0
        self.started_at = time.time()
        self.processing_durations: deque[float] = deque(maxlen=50)
        self.speed_test_done = 0
        self.speed_test_total = 0
        self.speed_test_step: int | None = None
        self.speed_test_mean_seconds: float | None = None

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.in_queue: queue.Queue = queue.Queue(maxsize=100)
        self.out_queue: queue.Queue = queue.Queue(maxsize=100)
        self.pending_target_label = target_label
        self.current_target_label = target_label
        self.style_states: dict[str, dict] = {}

        self.pa: pyaudio.PyAudio | None = None
        self.input_stream = None
        self.output_stream = None
        self.threads: list[threading.Thread] = []

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "status": self.status,
                "error": self.error,
                "slow_chunks": self.slow_chunks,
                "processed_chunks": self.processed_chunks,
                "runtime_steps": self.runtime_steps,
                "configured_steps": self.steps,
                "run_benchmark": self.run_benchmark,
                "autotuned_steps": self.autotuned_steps,
                "autotuned_version": self.autotuned_version,
                "current_target_label": self.current_target_label,
                "pending_target_label": self.pending_target_label,
                "available_target_labels": list(self.target_paths),
                "started_at": self.started_at,
                "chunk_size": self.chunk_size,
                "cfg_strength": self.cfg_strength,
            }

    def monitor_snapshot(self) -> dict:
        with self.lock:
            durations = list(self.processing_durations)
            budget_seconds = self.chunk_size / RATE
            mean_seconds = float(np.mean(durations)) if durations else None
            last_seconds = durations[-1] if durations else None
            max_seconds = float(np.max(durations)) if durations else None

            return {
                "status": self.status,
                "error": self.error,
                "slow_chunks": self.slow_chunks,
                "processed_chunks": self.processed_chunks,
                "runtime_steps": self.runtime_steps,
                "configured_steps": self.steps,
                "run_benchmark": self.run_benchmark,
                "autotuned_steps": self.autotuned_steps,
                "autotuned_version": self.autotuned_version,
                "cfg_strength": self.cfg_strength,
                "current_target_label": self.current_target_label,
                "pending_target_label": self.pending_target_label,
                "available_target_labels": list(self.target_paths),
                "started_at": self.started_at,
                "chunk_size": self.chunk_size,
                "budget_seconds": budget_seconds,
                "mean_seconds": mean_seconds,
                "last_seconds": last_seconds,
                "max_seconds": max_seconds,
                "input_queue": self.in_queue.qsize(),
                "output_queue": self.out_queue.qsize(),
                "speed_test_done": self.speed_test_done,
                "speed_test_total": self.speed_test_total,
                "speed_test_step": self.speed_test_step,
                "speed_test_mean_seconds": self.speed_test_mean_seconds,
            }

    def update_speed_test_progress(self, done: int, total: int, step: int, mean_seconds: float) -> None:
        with self.lock:
            self.speed_test_done = done
            self.speed_test_total = total
            self.speed_test_step = step
            self.speed_test_mean_seconds = mean_seconds

    def set_status(self, status: str) -> None:
        with self.lock:
            self.status = status

    def set_error(self, message: str) -> None:
        with self.lock:
            self.error = message
            self.status = "Failed"
        self.stop_event.set()
        self._poke_queues()

    def _poke_queues(self) -> None:
        for q in (self.in_queue, self.out_queue):
            try:
                q.put_nowait(None)
            except queue.Full:
                pass

    def request_target(self, label: str) -> None:
        with self.lock:
            if label not in self.target_paths:
                self.error = f"Target is not part of this streaming session: {label}"
                self.status = "Failed"
                self.stop_event.set()
                self._poke_queues()
                return
            if label == self.pending_target_label:
                return
            self.pending_target_label = label
            if label != self.current_target_label and label in self.style_states:
                self.status = "Switching target"

    def update_runtime(self, steps: int, cfg_strength: float) -> None:
        with self.lock:
            runtime_steps = max(SPEED_TEST_MIN_STEPS, int(steps))
            self.steps = runtime_steps
            self.runtime_steps = runtime_steps
            self.cfg_strength = float(cfg_strength)

    def is_alive(self) -> bool:
        return any(thread.is_alive() for thread in self.threads)

    def start(self) -> None:
        thread = threading.Thread(target=self._run, daemon=True)
        self.threads = [thread]
        thread.start()

    def stop(self) -> None:
        self.set_status("Stopping")
        self.stop_event.set()
        self._poke_queues()
        for thread in self.threads:
            thread.join(timeout=2.0)
        self._close_audio()
        self.set_status("Stopped")

    def _run(self) -> None:
        try:
            self.set_status("Loading model")
            model = load_streaming_model(str(CKPT_PATH), self.device)
            for idx, (label, path) in enumerate(self.target_paths.items(), start=1):
                self.set_status(f"Preparing target {idx}/{len(self.target_paths)}")
                style_state = prepare_style_state(model, path, self.chunk_size)
                with self.lock:
                    self.style_states[label] = style_state
            initial_style_state = self.style_states[self.target_label]

            if self.run_benchmark:
                self.set_status("Running speed test")
                recommended_steps, _ = run_speed_test(
                    model,
                    self.target_paths[self.target_label],
                    self.chunk_size,
                    SPEED_TEST_MAX_STEPS,
                    self.cfg_strength,
                    self.device,
                    progress_callback=self.update_speed_test_progress,
                )
                with self.lock:
                    self.steps = recommended_steps
                    self.runtime_steps = recommended_steps
                    self.autotuned_steps = recommended_steps
                    self.autotuned_version += 1

            crossfader = StreamingCrossfader(overlap=OVERLAP, device=self.device)
            apply_style_state(model, crossfader, initial_style_state)
            self._open_audio()
            self.set_status("Streaming")

            t_in = threading.Thread(target=self._input_loop, daemon=True)
            t_proc = threading.Thread(target=self._processing_loop, args=(model, crossfader), daemon=True)
            t_out = threading.Thread(target=self._output_loop, daemon=True)
            self.threads.extend([t_in, t_proc, t_out])
            t_in.start()
            t_proc.start()
            t_out.start()

            while not self.stop_event.is_set() and any(t.is_alive() for t in (t_in, t_proc, t_out)):
                time.sleep(0.1)
        except Exception as exc:
            self.set_error(f"Streaming failed: {exc}")
            traceback.print_exc()
        finally:
            self.stop_event.set()
            self._poke_queues()
            self._close_audio()
            with self.lock:
                if self.error is None and self.status != "Stopped":
                    self.status = "Stopped"

    def _open_audio(self) -> None:
        self.pa = pyaudio.PyAudio()
        self.input_stream = self.pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=self.input_device_index,
            frames_per_buffer=IO_CHUNK,
        )
        self.output_stream = self.pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True,
            output_device_index=self.output_device_index,
            frames_per_buffer=IO_CHUNK,
        )

    def _close_audio(self) -> None:
        for stream in (self.input_stream, self.output_stream):
            if stream is None:
                continue
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        self.input_stream = None
        self.output_stream = None
        if self.pa is not None:
            try:
                self.pa.terminate()
            except Exception:
                pass
        self.pa = None

    def _input_loop(self) -> None:
        try:
            while not self.stop_event.is_set():
                data = self.input_stream.read(IO_CHUNK, exception_on_overflow=False)
                x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                self.in_queue.put(x, timeout=0.5)
        except Exception as exc:
            if not self.stop_event.is_set():
                self.set_error(f"Input audio failed: {exc}")
        finally:
            try:
                self.in_queue.put_nowait(None)
            except queue.Full:
                pass

    def _processing_loop(self, model: CFMLightningModule, crossfader: StreamingCrossfader) -> None:
        buffer = np.zeros(0, dtype=np.float32)
        window_size = total_needed(self.chunk_size)
        chunk_duration = self.chunk_size / RATE
        monitor_slow_chunks = not self.run_benchmark

        try:
            while not self.stop_event.is_set():
                item = self.in_queue.get()
                if item is None:
                    break

                buffer = np.concatenate([buffer, item], axis=0)
                while buffer.shape[0] >= window_size and not self.stop_event.is_set():
                    window = buffer[:window_size].copy()
                    buffer = buffer[self.chunk_size:]

                    with self.lock:
                        switch_label = None
                        pending_label = self.pending_target_label
                        if pending_label != self.current_target_label:
                            self.current_target_label = pending_label
                            switch_label = pending_label
                            self.slow_chunks = 0
                            self.slow_monitor_grace_chunks = SLOW_CHUNK_LIMIT
                            self.status = "Switching target"
                        runtime_steps = self.runtime_steps
                        cfg_strength = self.cfg_strength

                    if switch_label is not None:
                        apply_style_state(model, crossfader, self.style_states[switch_label])
                        with self.lock:
                            self.status = "Streaming"

                    synchronize_device(model.device)
                    start_t = time.perf_counter()
                    out_wav = model.realtime_sample(
                        window,
                        steps=runtime_steps,
                        cfg_strength=cfg_strength,
                    )
                    out_block = crossfader.push(out_wav)
                    synchronize_device(model.device)
                    elapsed = time.perf_counter() - start_t

                    with self.lock:
                        self.processed_chunks += 1
                        self.processing_durations.append(float(elapsed))
                        if monitor_slow_chunks and self.slow_monitor_grace_chunks > 0:
                            self.slow_monitor_grace_chunks -= 1
                        elif monitor_slow_chunks and elapsed > chunk_duration:
                            self.slow_chunks += 1
                            if self.slow_chunks >= SLOW_CHUNK_LIMIT:
                                self.error = streamability_failure_message(
                                    elapsed,
                                    chunk_duration,
                                    self.slow_chunks,
                                )
                                print(f"[Streamability] {self.error}", flush=True)
                                self.status = "Failed"
                                self.stop_event.set()
                                self._poke_queues()
                                break
                        elif monitor_slow_chunks:
                            self.slow_chunks = 0

                    if out_block.numel() > 0:
                        y = out_block.detach().cpu().numpy().astype(np.float32)
                        self.out_queue.put(y, timeout=0.5)
        except Exception as exc:
            if not self.stop_event.is_set():
                self.set_error(f"Realtime processing failed: {exc}")
                traceback.print_exc()
        finally:
            try:
                tail = crossfader.flush()
                if tail.numel() > 0:
                    self.out_queue.put(tail.detach().cpu().numpy().astype(np.float32))
            except Exception:
                pass
            try:
                self.out_queue.put_nowait(None)
            except queue.Full:
                pass

    def _output_loop(self) -> None:
        try:
            while not self.stop_event.is_set():
                item = self.out_queue.get()
                if item is None:
                    break
                y = np.clip(np.asarray(item, dtype=np.float32), -1.0, 1.0)
                self.output_stream.write((y * 32767.0).astype(np.int16).tobytes())
        except Exception as exc:
            if not self.stop_event.is_set():
                self.set_error(f"Output audio failed: {exc}")


def ensure_controller_stopped() -> None:
    controller: StreamingController | None = st.session_state.get("streaming_controller")
    if controller is not None and controller.snapshot()["status"] == "Stopped":
        if not controller.is_alive():
            st.session_state["streaming_controller"] = None


def short_label(label: str | None) -> str:
    if not label:
        return "-"
    if "::" in label:
        label = label.split("::", 1)[1]
    parts = label.split("/")
    if len(parts) > 3:
        return "/".join(parts[-3:])
    return label


def format_ms(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    return f"{seconds * 1000:.0f} ms"


def render_compact_stat(column, label: str, value) -> None:
    column.caption(label)
    column.markdown(f"**{value}**")


def realtime_health(stats: dict) -> tuple[str, float]:
    mean_seconds = stats.get("mean_seconds")
    budget_seconds = stats.get("budget_seconds") or 1.0
    status = stats.get("status", "Idle")
    if status in {"Failed", "Stopped"}:
        return status, 0.0
    if mean_seconds is None:
        return "Waiting", 0.0
    ratio = mean_seconds / budget_seconds
    if ratio < 0.75:
        return "Healthy", ratio
    if ratio < 1.0:
        return "Near budget", ratio
    return "Behind realtime", ratio


def default_monitor_stats() -> dict:
    return {
        "status": "Idle",
        "runtime_steps": "-",
        "cfg_strength": "-",
        "chunk_size": 9600,
        "budget_seconds": 9600 / RATE,
        "processed_chunks": 0,
        "slow_chunks": 0,
        "current_target_label": None,
        "pending_target_label": None,
        "mean_seconds": None,
        "last_seconds": None,
        "max_seconds": None,
        "input_queue": 0,
        "output_queue": 0,
        "error": None,
        "speed_test_done": 0,
        "speed_test_total": 0,
        "speed_test_step": None,
        "speed_test_mean_seconds": None,
    }


def current_monitor_stats() -> dict:
    controller: StreamingController | None = st.session_state.get("streaming_controller")
    if controller is None:
        return default_monitor_stats()
    return controller.monitor_snapshot()


def render_live_monitor_stats(stats: dict | None = None) -> None:
    if stats is None:
        stats = current_monitor_stats()
    health, ratio = realtime_health(stats)

    header_cols = st.columns(6, vertical_alignment="top")
    render_compact_stat(header_cols[0], "Status", stats["status"])
    render_compact_stat(header_cols[1], "Health", health)
    render_compact_stat(header_cols[2], "Active target", short_label(stats.get("current_target_label")))
    render_compact_stat(header_cols[3], "Window", f"{stats['chunk_size'] / RATE:.2f}s")
    render_compact_stat(header_cols[4], "Steps", stats["runtime_steps"])
    render_compact_stat(header_cols[5], "CFG", stats["cfg_strength"])

    stat_cols = st.columns(6, vertical_alignment="top")
    render_compact_stat(stat_cols[0], "Chunks", stats["processed_chunks"])
    render_compact_stat(stat_cols[1], "Last", format_ms(stats.get("last_seconds")))
    render_compact_stat(stat_cols[2], "Mean", format_ms(stats.get("mean_seconds")))
    render_compact_stat(stat_cols[3], "Budget", format_ms(stats.get("budget_seconds")))
    render_compact_stat(stat_cols[4], "Slow", stats["slow_chunks"])
    render_compact_stat(stat_cols[5], "Queues", f"{stats['input_queue']} / {stats['output_queue']}")

    if stats.get("pending_target_label") != stats.get("current_target_label"):
        st.caption(f"Queued target: {short_label(stats.get('pending_target_label'))}")

    progress_text = (
        f"Mean chunk time {format_ms(stats.get('mean_seconds'))} / "
        f"budget {format_ms(stats.get('budget_seconds'))}"
    )
    st.progress(min(max(ratio, 0.0), 1.0), text=progress_text)

    if stats.get("error"):
        st.error(stats["error"])

    if stats["status"] == "Running speed test":
        done = stats.get("speed_test_done") or 0
        total = stats.get("speed_test_total") or 1
        step = stats.get("speed_test_step")
        mean_text = format_ms(stats.get("speed_test_mean_seconds"))
        label = f"Speed test {done}/{total}"
        if step is not None:
            label += f" · steps={step} · mean={mean_text}"
        st.progress(min(done / total, 1.0), text=label)


if hasattr(st, "fragment"):
    @st.fragment(run_every="0.5s")
    def render_live_monitor_stats_fragment() -> None:
        controller: StreamingController | None = st.session_state.get("streaming_controller")
        stats = current_monitor_stats()
        if sync_autotuned_steps_to_ui(controller, stats):
            st.rerun()
        render_live_monitor_stats(stats)
else:
    def render_live_monitor_stats_fragment() -> None:
        render_live_monitor_stats()


def render_live_monitor_body(
    *,
    show_controls: bool = False,
    start_disabled: bool = True,
    stop_disabled: bool = True,
    stop_label: str = "Stop",
    start_callback=None,
    start_kwargs: dict | None = None,
    stop_callback=None,
) -> tuple[bool, bool]:
    start_clicked = False
    stop_clicked = False

    with st.container(border=True):
        monitor_col, action_col = st.columns([7, 1], vertical_alignment="top")

        with monitor_col:
            if st.session_state.get("streaming_controller") is None:
                render_live_monitor_stats()
            else:
                render_live_monitor_stats_fragment()

        with action_col:
            st.caption("Controls")
            if show_controls:
                start_clicked = st.button(
                    "Start",
                    type="primary",
                    disabled=start_disabled,
                    key="top_start_streaming",
                    use_container_width=True,
                    on_click=start_callback,
                    kwargs=start_kwargs or {},
                )
                stop_clicked = st.button(
                    stop_label,
                    disabled=stop_disabled,
                    key="top_stop_streaming",
                    use_container_width=True,
                    on_click=stop_callback,
                )

    return start_clicked, stop_clicked


def render_live_monitor(**kwargs) -> tuple[bool, bool]:
    return render_live_monitor_body(**kwargs)


def initialize_streaming_widget_state() -> None:
    st.session_state.setdefault(STEPS_WIDGET_KEY, 10)
    if st.session_state[STEPS_WIDGET_KEY] < SPEED_TEST_MIN_STEPS:
        st.session_state[STEPS_WIDGET_KEY] = SPEED_TEST_MIN_STEPS
    st.session_state.setdefault(CFG_WIDGET_KEY, 2.0)
    st.session_state.setdefault(CHUNK_SIZE_WIDGET_KEY, 9600)


def sync_autotuned_steps_to_ui(controller: StreamingController | None, snapshot: dict) -> bool:
    if controller is None:
        return False
    autotuned_version = snapshot.get("autotuned_version", 0)
    autotuned_steps = snapshot.get("autotuned_steps")
    if not autotuned_version or autotuned_steps is None:
        return False

    sync_marker = (id(controller), autotuned_version)
    if st.session_state.get("streaming_autotune_sync_marker") == sync_marker:
        return False

    autotuned_steps = int(autotuned_steps)
    changed = st.session_state.get(STEPS_WIDGET_KEY) != autotuned_steps
    st.session_state[STEPS_WIDGET_KEY] = autotuned_steps
    st.session_state["streaming_autotune_sync_marker"] = sync_marker
    return changed


def default_streaming_target_labels(target_choices) -> list[str]:
    example_labels = [
        choice.label
        for choice in target_choices
        if choice.label.startswith("assets/target_examples/")
    ]
    if example_labels:
        return example_labels
    default_choice = target_choices[default_target_index(target_choices)]
    return [default_choice.label]


def default_initial_target_label(target_choices, selected_labels: list[str]) -> str | None:
    if not selected_labels:
        return None
    default_choice = target_choices[default_target_index(target_choices)]
    if default_choice.label in selected_labels:
        return default_choice.label
    return selected_labels[0]


def sync_active_target_widget(selected_labels: list[str], default_label: str | None) -> None:
    if not selected_labels:
        st.session_state.pop(ACTIVE_TARGET_WIDGET_KEY, None)
        return
    active_label = st.session_state.get(ACTIVE_TARGET_WIDGET_KEY)
    if active_label not in selected_labels:
        st.session_state[ACTIVE_TARGET_WIDGET_KEY] = default_label or selected_labels[0]


def stable_label_key(label: str) -> str:
    return hashlib.sha1(label.encode("utf-8")).hexdigest()[:12]


def initialize_prepared_target_state(target_choices) -> None:
    valid_labels = {choice.label for choice in target_choices}
    if PREPARED_TARGETS_KEY not in st.session_state:
        st.session_state[PREPARED_TARGETS_KEY] = default_streaming_target_labels(target_choices)
    st.session_state[PREPARED_TARGETS_KEY] = [
        label for label in st.session_state[PREPARED_TARGETS_KEY] if label in valid_labels
    ]


def add_prepared_target(label: str) -> None:
    selected_labels = st.session_state.setdefault(PREPARED_TARGETS_KEY, [])
    if label not in selected_labels:
        selected_labels.append(label)


def remove_prepared_target(label: str) -> None:
    st.session_state[PREPARED_TARGETS_KEY] = [
        selected_label
        for selected_label in st.session_state.get(PREPARED_TARGETS_KEY, [])
        if selected_label != label
    ]


def start_streaming_session(
    *,
    target_map: dict,
    input_device_index: int,
    output_device_index: int,
    device: str,
    chunk_size: int,
    steps: int,
    cfg_strength: float,
    run_benchmark: bool,
) -> None:
    if st.session_state.get("streaming_controller") is not None:
        return

    selected_labels = st.session_state.get(PREPARED_TARGETS_KEY, [])
    target_label = st.session_state.get(ACTIVE_TARGET_WIDGET_KEY)
    if target_label not in selected_labels:
        target_label = selected_labels[0] if selected_labels else None
    if target_label is None:
        return

    target_paths = {
        label: resolve_target_choice(target_map[label])
        for label in selected_labels
    }
    controller = StreamingController(
        target_label=target_label,
        target_paths=target_paths,
        input_device_index=input_device_index,
        output_device_index=output_device_index,
        device=device,
        chunk_size=chunk_size,
        steps=steps,
        cfg_strength=cfg_strength,
        run_benchmark=run_benchmark,
    )
    st.session_state["streaming_controller"] = controller
    controller.start()


def stop_streaming_session() -> None:
    controller: StreamingController | None = st.session_state.get("streaming_controller")
    if controller is not None:
        controller.stop()
    st.session_state["streaming_controller"] = None


def queue_active_target_change() -> None:
    controller: StreamingController | None = st.session_state.get("streaming_controller")
    target_label = st.session_state.get(ACTIVE_TARGET_WIDGET_KEY)
    if controller is not None and target_label is not None:
        controller.request_target(target_label)


def render_active_target_selectbox(
    *,
    selected_target_labels: list[str],
    target_choices,
    snapshot: dict,
    running: bool,
) -> None:
    active_label = snapshot.get("pending_target_label") if running else None
    default_label = (
        active_label
        if active_label in selected_target_labels
        else default_initial_target_label(target_choices, selected_target_labels)
    )
    sync_active_target_widget(selected_target_labels, default_label)
    st.selectbox(
        "Active target",
        selected_target_labels,
        key=ACTIVE_TARGET_WIDGET_KEY,
        on_change=queue_active_target_change if running else None,
    )


if hasattr(st, "fragment"):
    @st.fragment
    def render_active_target_selectbox_fragment(**kwargs) -> None:
        render_active_target_selectbox(**kwargs)
else:
    def render_active_target_selectbox_fragment(**kwargs) -> None:
        render_active_target_selectbox(**kwargs)


def render_prepared_target_rows(selected_labels: list[str], target_map: dict, removable: bool) -> None:
    for row_start in range(0, len(selected_labels), 2):
        row_labels = selected_labels[row_start:row_start + 2]
        row_cols = st.columns(2, gap="medium", vertical_alignment="top")
        for col, label in zip(row_cols, row_labels):
            target_path = resolve_target_choice(target_map[label])
            target_wav = packaged_file_from_dir(target_path, ".wav")
            with col:
                header_cols = st.columns([4, 1], vertical_alignment="top")
                header_cols[0].markdown(f"**{label}**")
                if removable:
                    header_cols[1].button(
                        "Remove",
                        key=f"remove_target_{stable_label_key(label)}",
                        use_container_width=True,
                        on_click=remove_prepared_target,
                        args=(label,),
                    )
                if target_wav is not None:
                    st.audio(str(target_wav))
        if row_start + 2 < len(selected_labels):
            st.divider()


def main() -> None:
    st.set_page_config(page_title="StyleStream Realtime", layout="wide")
    inject_page_styles()
    st.title("StyleStream Realtime")
    initialize_streaming_widget_state()

    ensure_controller_stopped()
    controller: StreamingController | None = st.session_state.get("streaming_controller")
    controller_status = controller.snapshot()["status"] if controller is not None else "Idle"
    running = controller is not None and controller_status not in {"Failed", "Stopped"}
    needs_clear = controller is not None and controller_status == "Failed"

    if not CKPT_PATH.exists():
        st.error(f"Missing checkpoint: {CKPT_PATH}")
        st.stop()

    input_devices = list_audio_devices(input_device=True)
    output_devices = list_audio_devices(input_device=False)
    if not input_devices or not output_devices:
        st.error("No usable input/output audio devices were found.")
        st.stop()

    with st.spinner("Indexing target speaker inventory..."):
        target_choices = find_target_choices()
    if not target_choices:
        st.error("No target folders with both .wav and .npy files were found.")
        st.stop()

    target_labels = [choice.label for choice in target_choices]
    target_map = {choice.label: choice for choice in target_choices}
    initialize_prepared_target_state(target_choices)

    snapshot = controller.snapshot() if controller is not None else {}
    running_target_labels = snapshot.get("available_target_labels", []) if controller is not None else []
    sync_autotuned_steps_to_ui(controller, snapshot)
    monitor_placeholder = st.empty()

    config_col, target_col = st.columns([0.78, 1.22])

    with config_col:
        with st.container(border=True):
            st.subheader("Audio IO")
            input_label = st.selectbox(
                "Input",
                [audio_device_label(device) for device in input_devices],
                disabled=running or needs_clear,
            )
            output_label = st.selectbox(
                "Output",
                [audio_device_label(device) for device in output_devices],
                disabled=running or needs_clear,
            )
            input_device_index = dict((audio_device_label(device), device["index"]) for device in input_devices)[input_label]
            output_device_index = dict((audio_device_label(device), device["index"]) for device in output_devices)[output_label]

        with st.container(border=True):
            st.subheader("Streaming")
            chunk_size = st.slider(
                "Chunk size",
                min_value=3200,
                max_value=16000,
                step=320,
                disabled=running or needs_clear,
                key=CHUNK_SIZE_WIDGET_KEY,
            )
            steps = st.slider(
                "Inference steps",
                min_value=SPEED_TEST_MIN_STEPS,
                max_value=32,
                step=1,
                key=STEPS_WIDGET_KEY,
            )
            cfg_strength = st.slider("CFG strength", min_value=0.0, max_value=5.0, step=0.1, key=CFG_WIDGET_KEY)
            run_benchmark = st.checkbox("Run speed test", value=True, disabled=running or needs_clear)
            st.caption("Speed test can auto-pick the highest steps that fit the realtime budget.")
            device = st.selectbox("Runtime device", available_devices(), disabled=running or needs_clear)

    with target_col:
        with st.container(border=True):
            st.subheader("Target Speakers")
            if running or needs_clear:
                selected_target_labels = running_target_labels
            else:
                selected_target_labels = st.session_state.get(PREPARED_TARGETS_KEY, [])

            if not selected_target_labels:
                st.error("Select at least one target speaker before streaming.")
                target_label = None
            else:
                render_active_target_selectbox_fragment(
                    selected_target_labels=selected_target_labels,
                    target_choices=target_choices,
                    snapshot=snapshot,
                    running=running,
                )
                target_label = st.session_state.get(ACTIVE_TARGET_WIDGET_KEY)

            if not running and not needs_clear:
                addable_labels = [label for label in target_labels if label not in selected_target_labels]
                if addable_labels:
                    add_col, button_col = st.columns([4, 0.9], vertical_alignment="bottom")
                    add_label = add_col.selectbox("Add target", addable_labels)
                    button_col.button(
                        "Add",
                        use_container_width=True,
                        on_click=add_prepared_target,
                        args=(add_label,),
                    )

            if selected_target_labels:
                st.caption("Prepared target speakers")
                render_prepared_target_rows(
                    selected_target_labels,
                    target_map,
                    removable=not running and not needs_clear,
                )

            if running and controller is not None and target_label is not None:
                controller.update_runtime(steps, cfg_strength)

    with monitor_placeholder.container():
        render_live_monitor(
            show_controls=True,
            start_disabled=running or needs_clear or not selected_target_labels,
            stop_disabled=controller is None,
            stop_label="Clear" if needs_clear else "Stop",
            start_callback=start_streaming_session,
            start_kwargs={
                "target_map": target_map,
                "input_device_index": input_device_index,
                "output_device_index": output_device_index,
                "device": device,
                "chunk_size": chunk_size,
                "steps": steps,
                "cfg_strength": cfg_strength,
                "run_benchmark": run_benchmark,
            },
            stop_callback=stop_streaming_session,
        )


if __name__ == "__main__":
    main()
