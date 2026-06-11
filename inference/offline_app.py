from __future__ import annotations

import io
import hashlib
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath

import librosa
import numpy as np
import soundfile as sf
import streamlit as st
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stylizer.cfm_lightning_module import CFMLightningModule


SAMPLE_RATE = 16000
ASSETS_DIR = REPO_ROOT / "assets"
TARGET_SPKRS_DIR = ASSETS_DIR / "target_spkrs"
TARGET_TAR_CACHE_DIR = REPO_ROOT / ".cache" / "target_spkrs"
SOURCE_TAR_CACHE_DIR = REPO_ROOT / ".cache" / "source_wavs"
CKPT_PATH = ASSETS_DIR / "ckpts" / "stylizer-no-style-enc.ckpt"
IGNORED_SOURCE_NAMES = {"converted.wav"}


@dataclass(frozen=True)
class SourceChoice:
    label: str
    path: Path | None = None
    tar_path: Path | None = None
    wav_member: str | None = None


@dataclass(frozen=True)
class TargetChoice:
    label: str
    path: Path | None = None
    tar_path: Path | None = None
    member_dir: str | None = None
    wav_member: str | None = None
    npy_member: str | None = None


def rel_label(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def find_asset_source_choices() -> list[SourceChoice]:
    choices = []
    if not ASSETS_DIR.exists():
        return choices
    for path in ASSETS_DIR.rglob("*.wav"):
        if path.name in IGNORED_SOURCE_NAMES:
            continue
        choices.append(SourceChoice(label=rel_label(path), path=path))
    return sorted(choices, key=lambda choice: choice.label)


def packaged_file_from_dir(package_dir: Path, suffix: str) -> Path | None:
    preferred = package_dir / f"{package_dir.name}{suffix}"
    if preferred.exists():
        return preferred

    matches = sorted(package_dir.glob(f"*{suffix}"))
    if len(matches) == 1:
        return matches[0]
    return None


def find_asset_target_choices() -> list[TargetChoice]:
    targets = []
    if not ASSETS_DIR.exists():
        return targets
    for path in ASSETS_DIR.rglob("*"):
        if not path.is_dir():
            continue
        if packaged_file_from_dir(path, ".wav") and packaged_file_from_dir(path, ".npy"):
            targets.append(TargetChoice(label=rel_label(path), path=path))
    return sorted(targets, key=lambda choice: choice.label)


def target_tar_paths() -> list[Path]:
    if not TARGET_SPKRS_DIR.exists():
        return []
    return sorted(TARGET_SPKRS_DIR.glob("*.tar"))


def tar_cache_root(tar_path: Path) -> Path:
    stat = tar_path.stat()
    return TARGET_TAR_CACHE_DIR / f"{tar_path.stem}-{stat.st_size}-{stat.st_mtime_ns}"


def source_tar_cache_root(tar_path: Path) -> Path:
    stat = tar_path.stat()
    return SOURCE_TAR_CACHE_DIR / f"{tar_path.stem}-{stat.st_size}-{stat.st_mtime_ns}"


def preferred_member(member_dir: str, suffix: str, members: list[str]) -> str:
    preferred = str(PurePosixPath(member_dir) / f"{PurePosixPath(member_dir).name}{suffix}")
    if preferred in members:
        return preferred
    return sorted(members)[0]


def tar_member_names(tar_path_str: str) -> list[str]:
    try:
        result = subprocess.run(
            ["tar", "-tf", tar_path_str],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        with tarfile.open(tar_path_str, "r:*") as tar:
            return tar.getnames()


@st.cache_data(show_spinner=False)
def index_tar_source_wavs(tar_path_str: str, tar_size: int, tar_mtime_ns: int) -> list[str]:
    del tar_size, tar_mtime_ns
    wav_members = []
    for member_name in tar_member_names(tar_path_str):
        member_path = PurePosixPath(member_name)
        if member_path.suffix.lower() != ".wav":
            continue
        if member_path.name in IGNORED_SOURCE_NAMES:
            continue
        wav_members.append(member_name)
    return sorted(wav_members)


@st.cache_data(show_spinner=False)
def index_tar_targets(tar_path_str: str, tar_size: int, tar_mtime_ns: int) -> list[tuple[str, str, str]]:
    del tar_size, tar_mtime_ns
    files_by_dir: dict[str, dict[str, list[str]]] = {}
    for member_name in tar_member_names(tar_path_str):
        member_path = PurePosixPath(member_name)
        suffix = member_path.suffix.lower()
        if suffix not in (".wav", ".npy"):
            continue
        member_dir = str(member_path.parent)
        files_by_dir.setdefault(member_dir, {}).setdefault(suffix, []).append(member_name)

    targets = []
    for member_dir, suffix_members in files_by_dir.items():
        if ".wav" not in suffix_members or ".npy" not in suffix_members:
            continue
        wav_member = preferred_member(member_dir, ".wav", suffix_members[".wav"])
        npy_member = preferred_member(member_dir, ".npy", suffix_members[".npy"])
        targets.append((member_dir, wav_member, npy_member))
    return sorted(targets, key=lambda item: item[0])


def find_tar_source_choices() -> list[SourceChoice]:
    choices = []
    for tar_path in target_tar_paths():
        stat = tar_path.stat()
        for wav_member in index_tar_source_wavs(str(tar_path), stat.st_size, stat.st_mtime_ns):
            choices.append(
                SourceChoice(
                    label=f"{rel_label(tar_path)}::{wav_member}",
                    tar_path=tar_path,
                    wav_member=wav_member,
                )
            )
    return choices


def find_source_choices() -> list[SourceChoice]:
    return sorted(find_asset_source_choices() + find_tar_source_choices(), key=lambda choice: choice.label)


def find_tar_target_choices() -> list[TargetChoice]:
    choices = []
    for tar_path in target_tar_paths():
        stat = tar_path.stat()
        for member_dir, wav_member, npy_member in index_tar_targets(str(tar_path), stat.st_size, stat.st_mtime_ns):
            choices.append(
                TargetChoice(
                    label=f"{rel_label(tar_path)}::{member_dir}",
                    tar_path=tar_path,
                    member_dir=member_dir,
                    wav_member=wav_member,
                    npy_member=npy_member,
                )
            )
    return choices


def find_target_choices() -> list[TargetChoice]:
    return sorted(find_asset_target_choices() + find_tar_target_choices(), key=lambda choice: choice.label)


def cache_dir_for_target_choice(choice: TargetChoice) -> Path:
    assert choice.tar_path is not None
    assert choice.member_dir is not None
    digest = hashlib.sha1(choice.member_dir.encode("utf-8")).hexdigest()[:12]
    leaf_name = PurePosixPath(choice.member_dir).name
    return tar_cache_root(choice.tar_path) / f"{leaf_name}-{digest}"


def cache_path_for_source_choice(choice: SourceChoice) -> Path:
    assert choice.tar_path is not None
    assert choice.wav_member is not None
    digest = hashlib.sha1(choice.wav_member.encode("utf-8")).hexdigest()[:12]
    member_name = PurePosixPath(choice.wav_member).name
    return source_tar_cache_root(choice.tar_path) / f"{digest}-{member_name}"


def extract_tar_source(choice: SourceChoice) -> Path:
    assert choice.tar_path is not None
    assert choice.wav_member is not None

    output_path = cache_path_for_source_choice(choice)
    if output_path.exists():
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(choice.tar_path, "r:*") as tar:
        for member in tar:
            if member.name != choice.wav_member:
                continue
            if member.issym() or member.islnk():
                raise RuntimeError(f"Unsafe tar member link: {member.name}")
            source = tar.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not read tar member: {member.name}")
            with output_path.open("wb") as output:
                shutil.copyfileobj(source, output)
            return output_path

    raise FileNotFoundError(f"Missing source wav in {choice.tar_path}: {choice.wav_member}")


def resolve_source_choice(choice: SourceChoice) -> Path:
    if choice.path is not None:
        return choice.path
    return extract_tar_source(choice)


def extract_tar_target(choice: TargetChoice) -> Path:
    assert choice.tar_path is not None
    assert choice.wav_member is not None
    assert choice.npy_member is not None

    package_dir = cache_dir_for_target_choice(choice)
    complete_marker = package_dir / ".complete"
    if complete_marker.exists() and packaged_file_from_dir(package_dir, ".wav") and packaged_file_from_dir(package_dir, ".npy"):
        return package_dir

    package_dir.mkdir(parents=True, exist_ok=True)
    wanted_members = {choice.wav_member, choice.npy_member}
    with tarfile.open(choice.tar_path, "r:*") as tar:
        for member in tar:
            if member.name not in wanted_members:
                continue
            if member.issym() or member.islnk():
                raise RuntimeError(f"Unsafe tar member link: {member.name}")
            source = tar.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not read tar member: {member.name}")
            output_path = package_dir / PurePosixPath(member.name).name
            with output_path.open("wb") as output:
                shutil.copyfileobj(source, output)
            wanted_members.remove(member.name)
            if not wanted_members:
                break

        if wanted_members:
            raise FileNotFoundError(f"Missing target files in {choice.tar_path}: {sorted(wanted_members)}")

    complete_marker.write_text("ok\n")
    return package_dir


def resolve_target_choice(choice: TargetChoice) -> Path:
    if choice.path is not None:
        return choice.path
    return extract_tar_target(choice)


def default_source_index(source_choices: list[SourceChoice]) -> int:
    for idx, choice in enumerate(source_choices):
        if choice.path is not None and choice.path.name == "source.wav":
            return idx
    for idx, choice in enumerate(source_choices):
        if choice.label.endswith("/source.wav") or choice.label.endswith("::source.wav"):
            return idx
    return 0


def default_target_index(target_choices: list[TargetChoice]) -> int:
    for idx, choice in enumerate(target_choices):
        if choice.path is None:
            continue
        wav_path = packaged_file_from_dir(choice.path, ".wav")
        if wav_path is not None and wav_path.name == "british.wav":
            return idx
    for idx, choice in enumerate(target_choices):
        if choice.label.endswith("/british"):
            return idx
    return 0


def standardize_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)
    if sample_rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    return np.clip(audio.astype(np.float32), -1.0, 1.0)


def read_audio_file(audio_file) -> np.ndarray:
    audio_file.seek(0)
    audio, sample_rate = sf.read(audio_file)
    return standardize_audio(audio, sample_rate)


def wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue()


def default_device() -> str:
    if sys.platform == "darwin" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def available_devices() -> list[str]:
    devices = [default_device()]
    for device in ("cuda", "mps", "cpu"):
        if device == "cuda" and not torch.cuda.is_available():
            continue
        if device == "mps" and not (
            sys.platform == "darwin"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            continue
        if device not in devices:
            devices.append(device)
    return devices


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str, device: str) -> CFMLightningModule:
    model = CFMLightningModule.load_for_inference(checkpoint_path, device=device)
    model.eval()
    return model


def main() -> None:
    st.set_page_config(page_title="StyleStream Offline", page_icon=None, layout="wide")
    st.title("StyleStream Offline Inference")

    with st.spinner("Indexing speaker packages..."):
        source_choices = find_source_choices()
        target_choices = find_target_choices()

    if not CKPT_PATH.exists():
        st.error(f"Missing checkpoint: {rel_label(CKPT_PATH)}")
        st.stop()
    if not target_choices:
        st.error("No target folders with both .wav and .npy files were found under assets/.")
        st.stop()

    source_input: str | np.ndarray | None = None

    source_col, _ = st.columns([1, 1])
    with source_col:
        st.subheader("Source")
        source_mode = st.radio("Source", ["Asset", "Recording"], horizontal=True)

        if source_mode == "Asset":
            if not source_choices:
                st.error("No source .wav files were found under assets/.")
                st.stop()
            source_label = st.selectbox(
                "Source audio",
                [choice.label for choice in source_choices],
                index=default_source_index(source_choices),
            )
            source_choice = {choice.label: choice for choice in source_choices}[source_label]
            with st.spinner("Loading selected source..."):
                source_path = resolve_source_choice(source_choice)
            source_input = str(source_path)
            st.audio(str(source_path))
        else:
            recording = st.audio_input("Source recording")
            if recording is not None:
                source_audio = read_audio_file(recording)
                source_input = source_audio
                st.audio(wav_bytes(source_audio), format="audio/wav")

    target_col, output_col = st.columns([1, 1])
    with target_col:
        st.subheader("Target")
        target_label = st.selectbox(
            "Target style",
            [choice.label for choice in target_choices],
            index=default_target_index(target_choices),
        )
        target_choice = {choice.label: choice for choice in target_choices}[target_label]
        with st.spinner("Loading selected target..."):
            target_path = resolve_target_choice(target_choice)
        target_wav = packaged_file_from_dir(target_path, ".wav")
        st.audio(str(target_wav))

        st.subheader("Settings")
        steps = st.slider("Steps", min_value=4, max_value=64, value=16, step=1)
        cfg_strength = st.slider("CFG strength", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
        device = st.selectbox("Device", available_devices())

        convert = st.button("Convert", type="primary", disabled=source_input is None)

    if convert:
        with st.spinner("Running inference..."):
            model = load_model(str(CKPT_PATH), device)
            out_dict = model.sample(
                cond=str(target_path),
                content=source_input,
                steps=int(steps),
                cfg_strength=float(cfg_strength),
            )
            pred_audio = out_dict["pred_audio"].squeeze().detach().cpu().numpy()
            pred_audio = np.clip(pred_audio.astype(np.float32), -1.0, 1.0)
            output_bytes = wav_bytes(pred_audio)
            st.session_state["generated_audio"] = output_bytes

    with output_col:
        st.subheader("Output")
        output_bytes = st.session_state.get("generated_audio")
        if output_bytes is None:
            st.info("No generated audio yet.")
        else:
            st.audio(output_bytes, format="audio/wav")
            st.download_button(
                "Download WAV",
                data=output_bytes,
                file_name="stylestream_converted.wav",
                mime="audio/wav",
            )


if __name__ == "__main__":
    main()
