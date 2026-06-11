from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.inference_offline import (  # noqa: E402
    SAMPLE_RATE,
    default_device,
    resolve_device,
    resolve_existing_path,
    resolve_source_arg,
    resolve_target_arg,
)
from inference.offline_app import CKPT_PATH  # noqa: E402
from stylizer.cfm_lightning_module import CFMLightningModule  # noqa: E402
from stylizer.modules import StreamingCrossfader  # noqa: E402


IO_CHUNK = 240
OVERLAP = 320 * 2
LOOKAHEAD = 80


def synchronize_device(device) -> None:
    device_type = getattr(device, "type", str(device)).split(":", 1)[0]
    if device_type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    elif device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def total_needed(chunk_size: int) -> int:
    return chunk_size + OVERLAP + LOOKAHEAD


def simulate_streaming(
    *,
    model: CFMLightningModule,
    source_audio: np.ndarray,
    target: str,
    chunk_size: int,
    steps: int,
    cfg_strength: float,
    device: str,
) -> tuple[np.ndarray, list[float]]:
    model.clear_cache()
    model.prepare_streaming(target, chunk_size)
    crossfader = StreamingCrossfader(overlap=OVERLAP, device=device)

    window_size = total_needed(chunk_size)
    padded_source = np.concatenate(
        [
            source_audio.astype(np.float32),
            np.zeros(window_size, dtype=np.float32),
        ],
        axis=0,
    )

    buffer = np.zeros(0, dtype=np.float32)
    output_blocks: list[np.ndarray] = []
    processing_times: list[float] = []
    budget_seconds = chunk_size / SAMPLE_RATE
    chunk_idx = 0

    for start in range(0, len(padded_source), IO_CHUNK):
        item = padded_source[start:start + IO_CHUNK]
        if item.shape[0] < IO_CHUNK:
            item = np.pad(item, (0, IO_CHUNK - item.shape[0]))
        buffer = np.concatenate([buffer, item.astype(np.float32)], axis=0)

        while buffer.shape[0] >= window_size:
            window = buffer[:window_size].copy()
            buffer = buffer[chunk_size:]

            synchronize_device(model.device)
            start_t = time.perf_counter()
            out_wav = model.realtime_sample(
                window,
                steps=steps,
                cfg_strength=cfg_strength,
            )
            out_block = crossfader.push(out_wav)
            synchronize_device(model.device)
            elapsed = time.perf_counter() - start_t
            processing_times.append(elapsed)
            chunk_idx += 1
            realtime_status = "realtime OK" if elapsed <= budget_seconds else "not realtime"
            print(
                f"[Chunk {chunk_idx:04d}] "
                f"input={chunk_size} samples ({budget_seconds * 1000:.1f} ms budget), "
                f"runtime={elapsed * 1000:.1f} ms -> {realtime_status}",
                flush=True,
            )

            if out_block.numel() > 0:
                output_blocks.append(out_block.detach().cpu().numpy().astype(np.float32))

    tail = crossfader.flush()
    if tail.numel() > 0:
        output_blocks.append(tail.detach().cpu().numpy().astype(np.float32))

    if output_blocks:
        audio = np.concatenate(output_blocks, axis=0)
    else:
        audio = np.zeros(0, dtype=np.float32)

    if audio.shape[0] < source_audio.shape[0]:
        audio = np.pad(audio, (0, source_audio.shape[0] - audio.shape[0]))
    audio = audio[:source_audio.shape[0]]
    return np.clip(audio.astype(np.float32), -1.0, 1.0), processing_times


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate StyleStream realtime inference on a full source wav.")
    parser.add_argument("--src", required=True, help="Source .wav path, source folder, or inventory label.")
    parser.add_argument("--tgt", required=True, help="Target style folder path or inventory label with .wav + .npy.")
    parser.add_argument("--steps", type=int, default=10, help="Realtime inference steps.")
    parser.add_argument("--cfg", type=float, default=2.0, help="Classifier-free guidance strength.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Runtime device.")
    parser.add_argument("--chunksize", type=int, default=9600, help="Streaming chunk advance in 16 kHz samples.")
    parser.add_argument("--ckpt", default=str(CKPT_PATH), help="Stylizer checkpoint path.")
    parser.add_argument("--out", default="inference/converted_simulated_streaming.wav", help="Output wav path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.chunksize <= 0 or args.chunksize % 320 != 0:
        raise ValueError("--chunksize must be a positive multiple of 320 samples.")

    device = resolve_device(args.device)
    source = resolve_source_arg(args.src)
    target = resolve_target_arg(args.tgt)
    ckpt_path = resolve_existing_path(args.ckpt) or Path(args.ckpt).expanduser()
    output_path = Path(args.out).expanduser()
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {ckpt_path}")
    print(f"Device: {device or default_device()}")
    model = CFMLightningModule.load_for_inference(str(ckpt_path), device=device)
    model.eval()

    print(f"Source: {source}")
    print(f"Target: {target}")
    print(f"Chunk size: {args.chunksize} samples ({args.chunksize / SAMPLE_RATE:.3f}s)")
    print(f"Window size: {total_needed(args.chunksize)} samples")
    source_audio = model.load_wav_reference(source)
    source_audio = np.asarray(source_audio, dtype=np.float32).reshape(-1)

    audio, processing_times = simulate_streaming(
        model=model,
        source_audio=source_audio,
        target=target,
        chunk_size=int(args.chunksize),
        steps=int(args.steps),
        cfg_strength=float(args.cfg),
        device=device,
    )
    sf.write(output_path, audio, SAMPLE_RATE)

    if processing_times:
        mean_ms = float(np.mean(processing_times)) * 1000.0
        max_ms = float(np.max(processing_times)) * 1000.0
        budget_ms = args.chunksize / SAMPLE_RATE * 1000.0
        print(
            f"Processed {len(processing_times)} chunks; "
            f"mean={mean_ms:.1f} ms, max={max_ms:.1f} ms, budget={budget_ms:.1f} ms"
        )
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
