"""
Download all data for the Biohub Cell Tracking During Development project.
Every source is skipped if its destination is already populated.
Requires kaggle oauth login.

Run with: uv run python src/data/download.py


Zebrahub raw imaging (ZSNS001.ome.zarr) is NOT downloaded: it is a zarr
directory tree with no listing API. Stream it lazily with zarr/ome-zarr:
    https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/ZSNS001.ome.zarr
"""

import os
import subprocess
import sys
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path("./data")
ARCHIVES_DIR = DATA_DIR / "_archives"

ZEBRAHUB_BASE = "https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective"
CTC_TRAIN = "https://data.celltrackingchallenge.net/training-datasets"


@dataclass(frozen=True)
class Source:
    key: str
    kind: str  # "git" | "kaggle-competition" | "kaggle-dataset" | "kaggle-kernel" | "http-zip" | "http-files"
    ref: str | list[str]  # git URL, kaggle slug, zip URL, or list of file URLs
    dest: Path


SOURCES: list[Source] = [
    # Official metric code, baseline model, GEFF<->CSV converters (always evaluate with this current metrics.py).
    Source(
        "organizer-repo",
        "git",
        "https://github.com/royerlab/kaggle-cell-tracking-competition",
        Path("./third_party") / "kaggle-cell-tracking-competition",
    ),
    # Competition train (zarr + GEFF) / test (zarr), ~88 GB. Requires accepted competition rules.
    Source("competition", "kaggle-competition", "biohub-cell-tracking-during-development", DATA_DIR / "competition"),
    # Public-frontier detector weights + offline wheels, 349 MB, CC0 ("50ep" actually holds the 402-epoch checkpoint).
    Source(
        "support-pack",
        "kaggle-dataset",
        "pilkwang/biohub-tracking-support-pack-50ep-v1",
        DATA_DIR / "support-pack",
    ),
    # Freitas synthetic 3D microscopy, ~18.5 GB, CC0 (165k labelled divisions). Published as kernel OUTPUT.
    Source("synthetic", "kaggle-kernel", "josefreitasalvesneto/biohub-synthetic-dataset", DATA_DIR / "synthetic"),
    # Zebrahub ZSNS001 tracks table (849 MB CSV) + zipped tracks benchmark, ~1 GB total.
    Source(
        "zebrahub",
        "http-files",
        [
            f"{ZEBRAHUB_BASE}/ZSNS001_tracks.csv",
            f"{ZEBRAHUB_BASE}/tracks_benchmark/tracks_int8_20230719.zarr.zip",
        ],
        DATA_DIR / "zebrahub",
    ),
    # Cell Tracking Challenge training sets: Drosophila light-sheet (5.8 GB), C. elegans (3.1 GB),
    # synthetic nuclei with perfect ground truth (3.1 GB).
    Source("ctc-dro", "http-zip", f"{CTC_TRAIN}/Fluo-N3DL-DRO.zip", DATA_DIR / "ctc" / "Fluo-N3DL-DRO"),
    Source("ctc-ce", "http-zip", f"{CTC_TRAIN}/Fluo-N3DH-CE.zip", DATA_DIR / "ctc" / "Fluo-N3DH-CE"),
    Source("ctc-sim", "http-zip", f"{CTC_TRAIN}/Fluo-N3DH-SIM+.zip", DATA_DIR / "ctc" / "Fluo-N3DH-SIM+"),
    # E. coli mother machine in CTC format, 315 MB, CC BY 4.0 -- for prototyping lineage/file handling.
    Source(
        "mother-machine",
        "http-zip",
        "https://zenodo.org/api/records/11237127/files/CTC.zip/content",
        DATA_DIR / "mother-machine",
    ),
    # Dense volumetric annotations for the CTC MDA231 sequences, 2.7 MB.
    Source(
        "cbia-mda231",
        "http-zip",
        "https://datasets.gryf.fi.muni.cz/isbi2025/Fluo-C3DL-MDA231_Full_Annotations.zip",
        DATA_DIR / "ctc" / "Fluo-C3DL-MDA231-full-annotations",
    ),
]


def _is_done(dest: Path) -> bool:
    return dest.exists() and (dest.is_file() or any(dest.iterdir()))


def _download_http(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-fL", "--retry", "3", "-C", "-", "-o", str(out), url], check=True)


def _extract_zip(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    archive.unlink()


def _kaggle_api() -> object:
    import kaggle

    kaggle.api.authenticate()
    return kaggle.api


def _handle_git(src: Source) -> None:
    assert isinstance(src.ref, str)
    src.dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", src.ref, str(src.dest)], check=True)


def _handle_kaggle_competition(src: Source) -> None:
    assert isinstance(src.ref, str)
    ARCHIVES_DIR.mkdir(parents=True, exist_ok=True)
    api = _kaggle_api()
    api.competition_download_files(src.ref, path=str(ARCHIVES_DIR), quiet=False)  # type: ignore[attr-defined]
    _extract_zip(ARCHIVES_DIR / f"{src.ref}.zip", src.dest)


def _handle_kaggle_dataset(src: Source) -> None:
    assert isinstance(src.ref, str)
    src.dest.mkdir(parents=True, exist_ok=True)
    api = _kaggle_api()
    api.dataset_download_files(src.ref, path=str(src.dest), unzip=True, quiet=False)  # type: ignore[attr-defined]


def _handle_kaggle_kernel(src: Source) -> None:
    assert isinstance(src.ref, str)
    src.dest.mkdir(parents=True, exist_ok=True)
    api = _kaggle_api()
    token: str | None = None
    while True:  # kernel outputs are paginated (~500 files here); follow tokens to the end
        _, token = api.kernels_output(  # type: ignore[attr-defined]
            src.ref, str(src.dest), quiet=False, page_token=token, page_size=100
        )
        if not token:
            break


def _handle_http_zip(src: Source) -> None:
    assert isinstance(src.ref, str)
    archive = ARCHIVES_DIR / f"{src.key}.zip"
    _download_http(src.ref, archive)
    _extract_zip(archive, src.dest)


def _handle_http_files(src: Source) -> None:
    assert isinstance(src.ref, list)
    for url in src.ref:
        _download_http(url, src.dest / url.rsplit("/", 1)[-1])


HANDLERS: dict[str, Callable[[Source], None]] = {
    "git": _handle_git,
    "kaggle-competition": _handle_kaggle_competition,
    "kaggle-dataset": _handle_kaggle_dataset,
    "kaggle-kernel": _handle_kaggle_kernel,
    "http-zip": _handle_http_zip,
    "http-files": _handle_http_files,
}


def main() -> None:
    kaggle_dir = Path.home() / ".kaggle"
    have_kaggle = bool(
        (kaggle_dir / "credentials.json").exists()  # kaggle 2.x OAuth (`kaggle auth login`)
        or (kaggle_dir / "kaggle.json").exists()  # legacy API token
        or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    )
    if not have_kaggle:
        print(
            "WARNING: no Kaggle credentials found. Run `kaggle auth login`, or place an API token at\n"
            "~/.kaggle/kaggle.json, or set KAGGLE_USERNAME/KAGGLE_KEY. Kaggle sources will be skipped.\n",
            file=sys.stderr,
        )

    for src in SOURCES:
        print(f"=== {src.key} -> {src.dest}")
        if src.kind.startswith("kaggle") and not have_kaggle:
            print("    skipped: no Kaggle credentials")
            continue
        if _is_done(src.dest):
            print("    skipped: already downloaded")
            continue
        HANDLERS[src.kind](src)
        print(f"    done: {src.key}")

    print("\nAll sources processed.")


if __name__ == "__main__":
    main()
