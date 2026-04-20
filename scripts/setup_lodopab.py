"""Configure dival to point at the LoDoPaB dataset and optionally download it.

Usage:
    python setup_lodopab.py --path /mnt/hdd/lodopab          # configure only
    python setup_lodopab.py --path /mnt/hdd/lodopab --download  # configure + download
"""
import argparse
import json
import os


DEFAULT_PATH = "/mnt/hdd/lodopab"
DIVAL_CONFIG = os.path.expanduser("~/.dival/config.json")


def set_dival_path(data_path):
    os.makedirs(os.path.dirname(DIVAL_CONFIG), exist_ok=True)
    config = {}
    if os.path.exists(DIVAL_CONFIG):
        with open(DIVAL_CONFIG) as f:
            config = json.load(f)
    config.setdefault("lodopab_dataset", {})["data_path"] = data_path
    with open(DIVAL_CONFIG, "w") as f:
        json.dump(config, f, indent=2)
    print(f"dival config updated: lodopab_dataset.data_path = {data_path}")


def download(data_path):
    try:
        from dival.datasets.lodopab_dataset import download_lodopab
        from dival.config import set_config
        set_config("lodopab_dataset/data_path", data_path)
        os.makedirs(data_path, exist_ok=True)
        print(f"Downloading LoDoPaB-CT to {data_path} (~150 GB)...")
        success = download_lodopab()
        if not success:
            print("Download failed. You can manually download from:")
            print("  https://zenodo.org/record/3384092")
            print(f"  and extract to: {data_path}")
    except Exception as e:
        print(f"Error during download: {e}")
        print("Manual download:")
        print("  https://zenodo.org/record/3384092")
        print(f"  Extract all HDF5 files to: {data_path}")


def check_files(data_path):
    expected = [
        "observation_train_000.hdf5",
        "ground_truth_train_000.hdf5",
        "observation_test_000.hdf5",
        "ground_truth_test_000.hdf5",
    ]
    found = [f for f in expected if os.path.exists(os.path.join(data_path, f))]
    missing = [f for f in expected if f not in found]
    if missing:
        print(f"Missing files in {data_path}:")
        for f in missing:
            print(f"  {f}")
    else:
        print(f"All expected files found in {data_path}.")
    return len(missing) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up LoDoPaB dataset for radfoam")
    parser.add_argument("--path", default=DEFAULT_PATH, help="Path to store LoDoPaB HDF5 files")
    parser.add_argument("--download", action="store_true", help="Download the dataset via dival")
    args = parser.parse_args()

    data_path = os.path.abspath(args.path)
    os.makedirs(data_path, exist_ok=True)

    set_dival_path(data_path)

    if args.download:
        download(data_path)
    else:
        print(f"Data path configured. Place LoDoPaB HDF5 files in: {data_path}")
        print("Or re-run with --download to fetch via dival (requires ~150 GB disk space).")
        print("Zenodo record: https://zenodo.org/record/3384092")

    check_files(data_path)
