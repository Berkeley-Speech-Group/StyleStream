from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.offline_app import (  # noqa: E402
    CKPT_PATH,
    find_source_choices,
    find_target_choices,
    resolve_source_choice,
    resolve_target_choice,
)
from stylizer.cfm_lightning_module import CFMLightningModule  # noqa: E402


SAMPLE_RATE = 16000


def default_device() -> str:
    if (
        sys.platform == "darwin"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_device(device: str) -> str:
    if device == "auto":
        return default_device()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    if device == "mps" and not (
        sys.platform == "darwin"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested, but it is not available on this machine.")
    return device


def resolve_existing_path(value: str) -> Path | None:
    path = Path(value).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(REPO_ROOT / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_source_arg(value: str) -> str:
    path = resolve_existing_path(value)
    if path is not None:
        return str(path)

    source_map = {choice.label: choice for choice in find_source_choices()}
    if value not in source_map:
        examples = "\n".join(f"  - {label}" for label in list(source_map)[:8])
        raise ValueError(f"Unknown source: {value}\nAvailable source labels include:\n{examples}")
    return str(resolve_source_choice(source_map[value]))


def resolve_target_arg(value: str) -> str:
    path = resolve_existing_path(value)
    if path is not None:
        return str(path)

    target_map = {choice.label: choice for choice in find_target_choices()}
    if value not in target_map:
        examples = "\n".join(f"  - {label}" for label in list(target_map)[:8])
        raise ValueError(f"Unknown target: {value}\nAvailable target labels include:\n{examples}")
    return str(resolve_target_choice(target_map[value]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline StyleStream inference.")
    parser.add_argument("--src", required=True, help="Source .wav path, source folder, or inventory label.")
    parser.add_argument("--tgt", required=True, help="Target style folder path or inventory label with .wav + .npy.")
    parser.add_argument("--steps", type=int, default=16, help="Flow matching inference steps.")
    parser.add_argument("--cfg", type=float, default=2.0, help="Classifier-free guidance strength.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Runtime device.")
    parser.add_argument("--ckpt", default=str(CKPT_PATH), help="Stylizer checkpoint path.")
    parser.add_argument("--out", default="inference/converted_offline.wav", help="Output wav path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")

    device = resolve_device(args.device)
    source = resolve_source_arg(args.src)
    target = resolve_target_arg(args.tgt)
    ckpt_path = resolve_existing_path(args.ckpt) or Path(args.ckpt).expanduser()
    output_path = Path(args.out).expanduser()
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {ckpt_path}")
    print(f"Device: {device}")
    model = CFMLightningModule.load_for_inference(str(ckpt_path), device=device)
    model.eval()

    print(f"Source: {source}")
    print(f"Target: {target}")
    out_dict = model.sample(
        cond=target,
        content=source,
        steps=int(args.steps),
        cfg_strength=float(args.cfg),
    )
    audio = out_dict["pred_audio"].squeeze().detach().cpu().numpy()
    audio = np.clip(audio.astype(np.float32), -1.0, 1.0)
    sf.write(output_path, audio, SAMPLE_RATE)
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
