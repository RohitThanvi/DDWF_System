#!/usr/bin/env python3
"""
Pull ERA5 reanalysis via the Copernicus Climate Data Store (CDS) API.

Prereqs:
  pip install cdsapi
  Create ~/.cdsapirc with your CDS API key (https://cds.climate.copernicus.eu/api-how-to)

Usage:
  python scripts/download_era5.py --year 2020 --month 01 --out ./data/raw

This pulls raw NetCDF; app/data/zarr_pipeline.py + scripts/build_zarr_store.py
turn it into the chunked Zarr store the rest of the system reads.
"""
from __future__ import annotations

import argparse
from pathlib import Path

PRESSURE_LEVEL_VARS = ["temperature", "u_component_of_wind", "v_component_of_wind",
                        "specific_humidity", "geopotential"]
SURFACE_VARS = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
                 "surface_pressure", "total_precipitation", "soil_temperature_level_1"]
PRESSURE_LEVELS = ["50", "100", "150", "200", "250", "300", "400", "500",
                    "600", "700", "850", "925", "1000"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", required=True)
    parser.add_argument("--month", required=True)
    parser.add_argument("--out", default="./data/raw")
    args = parser.parse_args()

    import cdsapi  # local import: only needed when this script actually runs

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    client.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "variable": PRESSURE_LEVEL_VARS,
            "pressure_level": PRESSURE_LEVELS,
            "year": args.year,
            "month": args.month,
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "format": "netcdf",
        },
        str(out_dir / f"era5_pl_{args.year}{args.month}.nc"),
    )

    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": SURFACE_VARS,
            "year": args.year,
            "month": args.month,
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "format": "netcdf",
        },
        str(out_dir / f"era5_sfc_{args.year}{args.month}.nc"),
    )


if __name__ == "__main__":
    main()
