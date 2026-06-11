import queue
import sys
import threading
import time
import traceback
import warnings
from copy import deepcopy
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
STYLE_ASSETS_DIR = REPO_ROOT / "assets" / "target_examples"

import numpy as np
import pyaudio
import torch

from stylizer.cfm_lightning_module import CFMLightningModule
from stylizer.modules import StreamingCrossfader


# ===========================
#  Audio / streaming config
# ===========================
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

IO_CHUNK = 240
CHUNK_SIZE = 9600
OVERLAP = 320 * 2
LOOKAHEAD = 80
TOTAL_NEEDED = CHUNK_SIZE + OVERLAP + LOOKAHEAD

INFERENCE_STEPS = 6 if sys.platform == "darwin" else 10
CFG_STRENGTH = 2.0

SPEED_TEST_ENABLED = True
SPEED_TEST_AUTO_USE_RECOMMENDED = True
SPEED_TEST_MAX_STEPS = 16
SPEED_TEST_MIN_STEPS = 6
SPEED_TEST_REPEATS = 3
SPEED_TEST_WARMUP_CHUNKS = 1
SPEED_TEST_BUDGET_SECONDS = 0.550
SPEED_TEST_MAX_CONTEXT_CHUNKS = (150 // ((TOTAL_NEEDED - LOOKAHEAD) // 320)) + 1

TGT_LIST = [
    str(STYLE_ASSETS_DIR / "hindi"),
    str(STYLE_ASSETS_DIR / "british"),
    str(STYLE_ASSETS_DIR / "arabic"),
    str(STYLE_ASSETS_DIR / "sad"),
    str(STYLE_ASSETS_DIR / "fear"),
    str(STYLE_ASSETS_DIR / "happy"),
    str(STYLE_ASSETS_DIR / "whisper"),
]


def mps_is_available():
    return (
        sys.platform == "darwin"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and hasattr(torch, "mps")
    )


def select_runtime_device():
    if mps_is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def synchronize_device(device):
    device_type = getattr(device, "type", str(device)).split(":", 1)[0]
    if device_type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    elif device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def snapshot_style_state(model):
    return {
        "tgt_content": model.tgt_content.clone(),
        "tgt_mel": model.tgt_mel.clone(),
        "ssl_style_emb": model.ssl_style_emb.clone(),
        "destylizer_buffer": deepcopy(model.destylizer_buffer),
    }


def apply_style_state(model, crossfader, style_state):
    if hasattr(model, "left_content"):
        del model.left_content
    model.vocoder.clear_cache()
    model.tgt_content = style_state["tgt_content"]
    model.tgt_mel = style_state["tgt_mel"]
    model.ssl_style_emb = style_state["ssl_style_emb"]
    model.destylizer_buffer = deepcopy(style_state["destylizer_buffer"])
    if crossfader is not None:
        crossfader.reset()


def preextract_style_states(model, target_paths):
    style_states = []
    print(f"Pre-extracting {len(target_paths)} styles...")
    for idx, target_path in enumerate(target_paths):
        print(f"  [{idx}] {target_path}")
        model.clear_cache()
        model.prepare_streaming(target_path, CHUNK_SIZE)
        style_states.append(snapshot_style_state(model))
    return style_states


def time_one_realtime_chunk(model, crossfader, chunk, steps, cfg_strength):
    synchronize_device(model.device)
    start_t = time.perf_counter()
    out_wav = model.realtime_sample(
        chunk,
        steps=steps,
        cfg_strength=cfg_strength,
    )
    crossfader.push(out_wav)
    synchronize_device(model.device)
    return time.perf_counter() - start_t


def prime_max_streaming_context(model, crossfader, chunk, cfg_strength):
    for _ in range(SPEED_TEST_MAX_CONTEXT_CHUNKS):
        time_one_realtime_chunk(model, crossfader, chunk, SPEED_TEST_MIN_STEPS, cfg_strength)


def run_speed_test(model, crossfader, initial_style_state, cfg_strength):
    print("\n[SpeedTest] Benchmarking realtime chunk processing before opening audio streams.")
    print(f"[SpeedTest] Chunk advance: {CHUNK_SIZE} samples ({CHUNK_SIZE / RATE:.3f}s)")
    print(f"[SpeedTest] Model input window: {TOTAL_NEEDED} samples")
    print(f"[SpeedTest] Timing at max retained source context: {SPEED_TEST_MAX_CONTEXT_CHUNKS} previous chunks")
    print(f"[SpeedTest] Budget: {SPEED_TEST_BUDGET_SECONDS * 1000:.0f} ms per chunk")

    rng = np.random.default_rng(0)
    test_chunk = rng.normal(0.0, 0.02, TOTAL_NEEDED).clip(-1.0, 1.0).astype(np.float32)
    results = []

    for steps in range(SPEED_TEST_MAX_STEPS, SPEED_TEST_MIN_STEPS - 1, -1):
        apply_style_state(model, crossfader, initial_style_state)
        prime_max_streaming_context(model, crossfader, test_chunk, cfg_strength)
        durations = []

        for repeat_idx in range(SPEED_TEST_WARMUP_CHUNKS + SPEED_TEST_REPEATS):
            elapsed = time_one_realtime_chunk(model, crossfader, test_chunk, steps, cfg_strength)
            if repeat_idx >= SPEED_TEST_WARMUP_CHUNKS:
                durations.append(elapsed)

        mean_seconds = float(np.mean(durations))
        results.append((steps, mean_seconds))
        status = "OK" if mean_seconds < SPEED_TEST_BUDGET_SECONDS else "slow"
        print(f"[SpeedTest] steps={steps:2d}: mean={mean_seconds * 1000:7.1f} ms  {status}")

    valid_steps = [steps for steps, mean_seconds in results if mean_seconds < SPEED_TEST_BUDGET_SECONDS]
    if valid_steps:
        recommended_steps = max(valid_steps)
        print(f"[SpeedTest] Recommended INFERENCE_STEPS = {recommended_steps}")
    else:
        apply_style_state(model, crossfader, initial_style_state)
        raise RuntimeError(
            "This machine cannot support streaming inference: even 6 inference steps "
            f"exceeded the realtime budget of {SPEED_TEST_BUDGET_SECONDS * 1000:.0f} ms per chunk. "
            "It is simply not fast enough for this streaming setup."
        )

    apply_style_state(model, crossfader, initial_style_state)
    return recommended_steps, results


# ===========================
#  Device selection
# ===========================
def find_16k_devices(p: pyaudio.PyAudio, input_device=True):
    """
    Return a list of (label_index, pyaudio_index, info) for devices
    that have input/output channels. We do not use is_format_supported
    because it can hide usable devices.
    """
    kind = "INPUT" if input_device else "OUTPUT"
    all_devs = []

    for dev_idx in range(p.get_device_count()):
        info = p.get_device_info_by_index(dev_idx)
        max_in = info.get("maxInputChannels", 0)
        max_out = info.get("maxOutputChannels", 0)

        if input_device and max_in == 0:
            continue
        if not input_device and max_out == 0:
            continue

        all_devs.append((dev_idx, info))

    candidates = []
    label_idx = 0
    print(f"=== {kind} device candidates ===")

    for dev_idx, info in all_devs:
        name = info.get("name", "Unknown")
        default_sr = info.get("defaultSampleRate", "??")

        if input_device:
            in_ch = info.get("maxInputChannels", 0)
            print(f"[{label_idx}] {name} (In: {in_ch}, Default SR: {default_sr}, PyAudio #{dev_idx})")
        else:
            out_ch = info.get("maxOutputChannels", 0)
            print(f"[{label_idx}] {name} (Out: {out_ch}, Default SR: {default_sr}, PyAudio #{dev_idx})")

        candidates.append((label_idx, dev_idx, info))
        label_idx += 1

    if not candidates:
        print(f"No {kind} devices found at all.")

    return candidates


def select_16k_device(p: pyaudio.PyAudio, input_device=True) -> int:
    """
    Show available devices and let user choose.
    Returns the PyAudio device index.
    """
    candidates = find_16k_devices(p, input_device=input_device)
    if not candidates:
        raise RuntimeError("No audio devices available.")

    kind = "input" if input_device else "output"

    while True:
        try:
            idx = int(input(f"Select {kind} device by number: "))
            for label_idx, pa_idx, _ in candidates:
                if label_idx == idx:
                    return pa_idx
            print("Out of range, try again.")
        except Exception as e:
            print(f"Invalid selection ({e}), try again.")


# ===========================
#  Threads: input / processing / output
# ===========================
def input_thread_fn(stop_event, in_queue, input_stream):
    """
    Read audio from mic in small IO_CHUNK blocks, convert to float32 [-1,1],
    and push into in_queue.
    """
    try:
        while not stop_event.is_set():
            data = input_stream.read(IO_CHUNK, exception_on_overflow=False)
            x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            in_queue.put(x)
    except Exception as e:
        print(f"[InputThread] Exception: {e}")
    finally:
        in_queue.put(None)
        print("[InputThread] Stopped.")


def processing_thread_fn(
    stop_event,
    in_queue,
    out_queue,
    model,
    crossfader,
    steps: int = 10,
    cfg_strength: float = 2.0,
    style_states=None,
    style_selection=None,
    style_lock=None,
):
    """
    Accumulate IO_CHUNK blocks, process full model windows, apply queued style
    switches at chunk boundaries, and push converted audio into out_queue.
    """
    buffer = np.zeros(0, dtype=np.float32)

    try:
        while not stop_event.is_set():
            item = in_queue.get()
            if item is None:
                break

            buffer = np.concatenate([buffer, item], axis=0)

            while buffer.shape[0] >= TOTAL_NEEDED:
                window = buffer[:TOTAL_NEEDED].copy()
                buffer = buffer[CHUNK_SIZE:]

                if style_states is not None and style_selection is not None and style_lock is not None:
                    switch_idx = None
                    with style_lock:
                        pending_idx = style_selection["pending"]
                        if pending_idx != style_selection["current"]:
                            style_selection["current"] = pending_idx
                            switch_idx = pending_idx
                    if switch_idx is not None:
                        print(f"[StyleSwitch] Applying style #{switch_idx} at chunk boundary.")
                        apply_style_state(model, crossfader, style_states[switch_idx])

                synchronize_device(model.device)
                start_t = time.time()
                out_wav = model.realtime_sample(
                    window,
                    steps=steps,
                    cfg_strength=cfg_strength,
                )
                out_block = crossfader.push(out_wav)
                synchronize_device(model.device)
                end_t = time.time()

                print(
                    f"[ProcessThread] processed {CHUNK_SIZE / RATE:.3f}s chunk "
                    f"in {end_t - start_t:.4f}s, emitted {out_block.numel()} samples"
                )

                if out_block.numel() > 0:
                    y = out_block.detach().cpu().numpy().astype(np.float32)
                    out_queue.put(y)

    except Exception as e:
        print(f"[ProcessThread] Exception: {e}")
        traceback.print_exc()
    finally:
        tail = crossfader.flush()
        if tail.numel() > 0:
            y = tail.detach().cpu().numpy().astype(np.float32)
            out_queue.put(y)

        out_queue.put(None)
        print("[ProcessThread] Stopped.")


def output_thread_fn(stop_event, out_queue, output_stream):
    """
    Take float32 blocks from out_queue, convert to int16, and write to speaker.
    """
    try:
        while not stop_event.is_set():
            item = out_queue.get()
            if item is None:
                break

            y = np.asarray(item, dtype=np.float32)
            y = np.clip(y, -1.0, 1.0)
            y_int16 = (y * 32767.0).astype(np.int16)
            output_stream.write(y_int16.tobytes())
    except Exception as e:
        print(f"[OutputThread] Exception: {e}")
    finally:
        print("[OutputThread] Stopped.")


# ===========================
#  Style switch input
# ===========================
def print_style_menu(num_styles, style_list=None):
    print(f"\n{'=' * 60}")
    print(f"[StyleSwitch] READY - type a number [0..{num_styles - 1}] + Enter to switch")
    print("[StyleSwitch] Available styles:")
    for i in range(num_styles):
        style_name = style_list[i] if style_list else f"Style {i}"
        print(f"[StyleSwitch]   {i}: {style_name}")
    print(f"{'=' * 60}\n")


def start_windows_style_index_listener(stop_event, num_styles, on_select, style_list=None):
    import msvcrt

    def _listener():
        buf = ""
        print_style_menu(num_styles, style_list)
        print(f"[StyleSwitch] Enter style index [0..{num_styles - 1}]: ", end="", flush=True)

        while not stop_event.is_set():
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\r", "\n"):
                    s = buf.strip()
                    buf = ""
                    if not s:
                        print(f"\n[StyleSwitch] Enter style index [0..{num_styles - 1}]: ", end="", flush=True)
                        continue
                    try:
                        idx = int(s)
                        if 0 <= idx < num_styles:
                            print(f"\n[StyleSwitch] Queueing style #{idx} ...")
                            on_select(idx)
                        else:
                            print(f"\n[StyleSwitch] Index out of range (0..{num_styles - 1}).")
                    except ValueError:
                        print("\n[StyleSwitch] Invalid index. Please enter an integer.")
                    print(f"[StyleSwitch] Enter style index [0..{num_styles - 1}]: ", end="", flush=True)
                elif ch in ("\b", "\x7f"):
                    if buf:
                        buf = buf[:-1]
                        print("\b \b", end="", flush=True)
                elif ch.isdigit():
                    buf += ch
                    print(ch, end="", flush=True)
            else:
                time.sleep(0.02)

    t = threading.Thread(target=_listener, daemon=True)
    t.start()
    return t


def start_stdin_style_index_listener(stop_event, num_styles, on_select, style_list=None):
    def _listener():
        print_style_menu(num_styles, style_list)

        while not stop_event.is_set():
            try:
                print(f"[StyleSwitch] Enter style index [0..{num_styles - 1}]: ", end="", flush=True)
                line = sys.stdin.readline().strip()

                if not line:
                    continue
                if stop_event.is_set():
                    break

                try:
                    idx = int(line)
                    if 0 <= idx < num_styles:
                        on_select(idx)
                    else:
                        print(f"[StyleSwitch] ERROR: Index {idx} out of range (must be 0..{num_styles - 1})\n")
                except ValueError:
                    print(f"[StyleSwitch] ERROR: '{line}' is not a valid number\n")

            except (EOFError, KeyboardInterrupt):
                print("\n[StyleSwitch] Input listener stopped.")
                stop_event.set()
                break
            except Exception as e:
                print(f"\n[StyleSwitch] Error: {e}")

    t = threading.Thread(target=_listener, daemon=True)
    t.start()
    return t


def start_style_index_listener(stop_event, num_styles, on_select, style_list=None):
    if sys.platform.startswith("win"):
        return start_windows_style_index_listener(stop_event, num_styles, on_select, style_list)
    return start_stdin_style_index_listener(stop_event, num_styles, on_select, style_list)


# ===========================
#  Main
# ===========================
def main():
    device = select_runtime_device()
    ckpt_path = str(REPO_ROOT / "assets" / "ckpts" / "stylizer-no-style-enc.ckpt")

    print(f"Platform: {sys.platform}")
    print(f"Runtime device: {device}")
    print("Loading model...")
    model = CFMLightningModule.load_for_inference(ckpt_path, device=device)
    model.eval()
    style_states = preextract_style_states(model, TGT_LIST)

    crossfader = StreamingCrossfader(overlap=OVERLAP, device=device)
    apply_style_state(model, crossfader, style_states[0])

    runtime_steps = INFERENCE_STEPS
    if SPEED_TEST_ENABLED:
        recommended_steps, _ = run_speed_test(model, crossfader, style_states[0], CFG_STRENGTH)
        if SPEED_TEST_AUTO_USE_RECOMMENDED:
            runtime_steps = recommended_steps
            print(f"[SpeedTest] Using {runtime_steps} steps for this streaming session.")
        elif recommended_steps != INFERENCE_STEPS:
            print(
                f"[SpeedTest] Current INFERENCE_STEPS={INFERENCE_STEPS}; "
                f"recommended={recommended_steps}. Edit the config if desired."
            )

    style_lock = threading.Lock()
    style_selection = {"current": 0, "pending": 0}

    p = pyaudio.PyAudio()

    print("\nSelect INPUT (microphone) device:")
    input_idx = select_16k_device(p, input_device=True)

    print("\nSelect OUTPUT (speaker/headphones) device:")
    output_idx = select_16k_device(p, input_device=False)

    print(f"\nUsing fixed sample rate: {RATE} Hz")
    print(f"Input PyAudio device index:  {input_idx}")
    print(f"Output PyAudio device index: {output_idx}")

    input_stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=input_idx,
        frames_per_buffer=IO_CHUNK,
    )

    output_stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        output=True,
        output_device_index=output_idx,
        frames_per_buffer=IO_CHUNK,
    )

    in_queue = queue.Queue(maxsize=100)
    out_queue = queue.Queue(maxsize=100)
    stop_event = threading.Event()

    t_in = threading.Thread(
        target=input_thread_fn,
        args=(stop_event, in_queue, input_stream),
        daemon=True,
    )
    t_proc = threading.Thread(
        target=processing_thread_fn,
        args=(
            stop_event,
            in_queue,
            out_queue,
            model,
            crossfader,
            runtime_steps,
            CFG_STRENGTH,
            style_states,
            style_selection,
            style_lock,
        ),
        daemon=True,
    )
    t_out = threading.Thread(
        target=output_thread_fn,
        args=(stop_event, out_queue, output_stream),
        daemon=True,
    )

    def apply_style(idx):
        with style_lock:
            style_selection["pending"] = idx
        print(f"\n[StyleSwitch] Queued style #{idx}; it will apply at the next chunk boundary.")

    start_style_index_listener(stop_event, len(TGT_LIST), apply_style, TGT_LIST)

    t_in.start()
    t_proc.start()
    t_out.start()

    print("\nAsync 3-thread conversion running.")
    print(f"  IO_CHUNK    = {IO_CHUNK} samples (~{IO_CHUNK / RATE * 1000:.1f} ms)")
    print(f"  CHUNK_SIZE  = {CHUNK_SIZE} samples (~{CHUNK_SIZE / RATE:.3f} s)")
    print(f"  STEPS       = {runtime_steps}")
    print(f"  CFG         = {CFG_STRENGTH}")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt, stopping...")
    finally:
        stop_event.set()

        try:
            in_queue.put_nowait(None)
        except queue.Full:
            pass
        try:
            out_queue.put_nowait(None)
        except queue.Full:
            pass

        t_in.join(timeout=2.0)
        t_proc.join(timeout=2.0)
        t_out.join(timeout=2.0)

        input_stream.stop_stream()
        input_stream.close()
        output_stream.stop_stream()
        output_stream.close()
        p.terminate()
        print("[Main] Clean exit.")


if __name__ == "__main__":
    main()
