"""Download and extract all LoDoPaB-CT splits from Zenodo record 3384092.

Writes progress to /mnt/hdd/lodopab/download.log
Run: micromamba run -n radfoam python download_lodopab.py
"""
import os
import sys
import json
import logging
from zipfile import ZipFile

DATA_PATH = "/mnt/hdd/lodopab"
ZENODO_RECORD = "3384092"
DIVAL_CONFIG = os.path.expanduser("~/.dival/config.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(DATA_PATH, "download.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger()

os.makedirs(DATA_PATH, exist_ok=True)

# Update dival config
cfg = {}
if os.path.exists(DIVAL_CONFIG):
    with open(DIVAL_CONFIG) as f:
        cfg = json.load(f)
cfg.setdefault("lodopab_dataset", {})["data_path"] = DATA_PATH
with open(DIVAL_CONFIG, "w") as f:
    json.dump(cfg, f, indent=2)
log.info(f"dival config set: {DATA_PATH}")

from dival.util.zenodo_download import download_zenodo_record

log.info("Starting download from Zenodo record 3384092 (~55 GB compressed)...")
success = download_zenodo_record(ZENODO_RECORD, DATA_PATH, auto_yes=True)

if not success:
    log.error("Download failed or checksum mismatch. Check download.log and retry.")
    sys.exit(1)

log.info("Download complete. Extracting zip files...")

zip_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".zip")]
for zf in sorted(zip_files):
    path = os.path.join(DATA_PATH, zf)
    log.info(f"Extracting {zf} ...")
    with ZipFile(path, "r") as z:
        z.extractall(DATA_PATH)
    os.remove(path)
    log.info(f"  Done, removed {zf}")

log.info("All done. LoDoPaB-CT is ready in " + DATA_PATH)
