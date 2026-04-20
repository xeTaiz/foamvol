"""Download and extract all 2DeteCT parts from Zenodo.

Progress logged to /mnt/hdd/2detectct/download.log
Run: micromamba run -n radfoam python download_2detectct.py
"""
import os
import sys
import logging
import requests
from zipfile import ZipFile
from tqdm import tqdm

DATA_PATH = "/mnt/hdd/2detectct"
os.makedirs(DATA_PATH, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(DATA_PATH, "download.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger()

FILES = [
    ("8014758", "2DeteCT_slices1-1000.zip",
     "https://zenodo.org/api/records/8014758/files/2DeteCT_slices1-1000.zip/content"),
    ("8014765", "2DeteCT_slices1001-2000.zip",
     "https://zenodo.org/api/records/8014766/files/2DeteCT_slices1001-2000.zip/content"),
    ("8017612", "2DeteCT_slices2001-3000_RecSeg.zip",
     "https://zenodo.org/api/records/8017612/files/2DeteCT_slices2001-3000_RecSeg.zip/content"),
    ("8014829", "2DeteCT_slices3001-4000.zip",
     "https://zenodo.org/api/records/8014829/files/2DeteCT_slices3001-4000.zip/content"),
    ("8014874", "2DeteCT_slices4001-5000.zip",
     "https://zenodo.org/api/records/8014874/files/2DeteCT_slices4001-5000.zip/content"),
    ("8014907", "2DeteCT_slicesOOD.zip",
     "https://zenodo.org/api/records/8014907/files/2DeteCT_slicesOOD.zip/content"),
]

CHUNK = 1 << 20  # 1 MB chunks


def download_file(url, dest):
    tmp = dest + ".part"
    existing = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}

    r = requests.get(url, headers=headers, stream=True, timeout=60)
    total = int(r.headers.get("content-length", 0)) + existing

    mode = "ab" if existing else "wb"
    with open(tmp, mode) as f, tqdm(
        total=total, initial=existing, unit="B", unit_scale=True, desc=os.path.basename(dest)
    ) as bar:
        for chunk in r.iter_content(CHUNK):
            f.write(chunk)
            bar.update(len(chunk))

    os.rename(tmp, dest)


for record_id, filename, url in FILES:
    dest = os.path.join(DATA_PATH, filename)
    if os.path.exists(dest):
        log.info(f"Already downloaded: {filename}")
    else:
        log.info(f"Downloading {filename} ...")
        try:
            download_file(url, dest)
            log.info(f"  Done: {filename}")
        except Exception as e:
            log.error(f"  Failed: {filename} — {e}")
            sys.exit(1)

log.info("All downloads complete. Extracting...")

for _, filename, _ in FILES:
    zip_path = os.path.join(DATA_PATH, filename)
    log.info(f"Extracting {filename} ...")
    with ZipFile(zip_path, "r") as z:
        z.extractall(DATA_PATH)
    os.remove(zip_path)
    log.info(f"  Extracted and removed {filename}")

log.info(f"Done. 2DeteCT ready in {DATA_PATH}")
