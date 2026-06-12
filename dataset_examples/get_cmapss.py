"""Download NASA C-MAPSS turbofan engine degradation training data.

Fetches the FD001 and FD002 training sets of the C-MAPSS (Commercial
Modular Aero-Propulsion System Simulation) turbofan engine degradation
dataset from a public GitHub mirror of the official NASA distribution.

Each file is space-separated with 26 columns:
    unit, cycle, op_setting_1..3, sensor_1..21
train_FD001.txt: 100 units, 20,631 rows (single operating condition).
train_FD002.txt: 260 units, 53,759 rows (six operating conditions).

Data source
-----------
Original: NASA Prognostics Data Repository (Prognostics Center of
Excellence, NASA Ames Research Center).
Mirror used: https://github.com/edwardzjl/CMAPSSData

Citation
--------
A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation
Data Set", NASA Prognostics Data Repository, NASA Ames Research Center,
Moffett Field, CA.

Usage
-----
    python get_cmapss.py [--out-dir DIR] [--force]
"""

import argparse
import os
import sys
import urllib.request

BASE_URL = "https://raw.githubusercontent.com/edwardzjl/CMAPSSData/master"
FILES = ["train_FD001.txt", "train_FD002.txt"]
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cmapss")


def download(url, dest):
    print(f"Downloading {url}\n  -> {dest}")
    with urllib.request.urlopen(url) as response, open(dest, "wb") as fh:
        fh.write(response.read())
    size_mb = os.path.getsize(dest) / 1e6
    print(f"  done ({size_mb:.1f} MB)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the files already exist",
    )
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    for name in FILES:
        dest = os.path.join(args.out_dir, name)
        if os.path.exists(dest) and not args.force:
            print(f"Skipping {name}: already exists (use --force to re-download)")
            continue
        download(f"{BASE_URL}/{name}", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
