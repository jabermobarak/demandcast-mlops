"""Acquire only the public UCI dataset; never execute files from the archive."""

import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATASET_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset"
DATASET_DOI = "10.24432/C5W894"
EXPECTED_SHA256 = "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401"


def download_dataset(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "bike-sharing.zip"
    if target.exists():
        payload = target.read_bytes()
    else:
        request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "DemandCast/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read(5_000_001)
        if len(payload) > 5_000_000:
            raise ValueError("Source archive exceeds the expected download limit")
    checksum = hashlib.sha256(payload).hexdigest()
    if EXPECTED_SHA256 and checksum != EXPECTED_SHA256:
        raise ValueError("UCI archive checksum mismatch; review the source before updating the pin")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        info = archive.getinfo("hour.csv")
        if info.file_size > 10_000_000:
            raise ValueError("Source CSV exceeds the expected size")
        raw = archive.read("hour.csv")
    target.write_bytes(payload)
    csv_path = directory / "hour.csv"
    csv_path.write_bytes(raw)
    provenance = {"url": DATASET_URL, "dataset_page": DATASET_PAGE, "doi": DATASET_DOI,
                  "author": "Hadi Fanaee-T", "year": 2013, "license": "CC BY 4.0",
                  "archive_sha256": checksum, "csv_sha256": hashlib.sha256(raw).hexdigest()}
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return csv_path, provenance


def read_observations(path):
    raw = pd.read_csv(path)
    required = {"dteday", "hr", "cnt"}
    if not required.issubset(raw):
        raise ValueError("Expected UCI hourly columns: dteday, hr, cnt")
    hours = pd.to_numeric(raw["hr"], errors="raise")
    if not hours.between(0, 23).all() or (hours % 1 != 0).any():
        raise ValueError("Hour must be an integer from 0 to 23")
    result = pd.DataFrame({
        "ts": pd.to_datetime(raw["dteday"], format="%Y-%m-%d") + pd.to_timedelta(hours, unit="h"),
        "y": pd.to_numeric(raw["cnt"], errors="raise").astype(float),
    })
    return validate_observations(result)


def validate_observations(frame):
    frame = frame[["ts", "y"]].copy()
    frame["ts"] = pd.to_datetime(frame["ts"])
    if frame.empty or frame["ts"].isna().any() or frame["ts"].duplicated().any():
        raise ValueError("Observations need unique, non-null hourly timestamps")
    if frame["ts"].dt.tz is not None:
        raise ValueError("This dataset uses timezone-naive timestamps")
    if (frame["ts"] != frame["ts"].dt.floor("h")).any():
        raise ValueError("Observations must lie on exact hours")
    if frame["y"].isna().any() or not frame["y"].between(0, 1_000_000).all():
        raise ValueError("Observed counts must be finite, non-negative numbers")
    return frame.sort_values("ts").reset_index(drop=True)
