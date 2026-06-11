"""Download the OPSD Germany daily electricity time series.

Fetches `opsd_germany_daily.csv`: daily electricity consumption, wind and
solar power production for Germany (2006-2017), derived from Open Power
System Data (OPSD).

Columns: Date, Consumption, Wind, Solar, Wind+Solar (GWh per day).

Data source
-----------
Original: Open Power System Data (OPSD), https://open-power-system-data.org/
Mirror used: https://github.com/jenfly/opsd (Jennifer Walker's processed
daily aggregation of the OPSD "Time series" package).

Usage
-----
    python get_opsd_germany.py [--out-dir DIR] [--force]
"""

import argparse
import os
import sys
import urllib.request

URL = "https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv"
FILENAME = "opsd_germany_daily.csv"
DEFAULT_OUT_DIR = os.path.dirname(os.path.abspath(__file__))


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
        help="Re-download even if the file already exists",
    )
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    dest = os.path.join(args.out_dir, FILENAME)
    if os.path.exists(dest) and not args.force:
        print(f"Skipping {FILENAME}: already exists (use --force to re-download)")
        return 0

    print(f"Downloading {URL}\n  -> {dest}")
    with urllib.request.urlopen(URL) as response, open(dest, "wb") as fh:
        fh.write(response.read())
    size_kb = os.path.getsize(dest) / 1e3
    print(f"  done ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
