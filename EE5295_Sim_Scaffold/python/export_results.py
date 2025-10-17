"""
Export simulation CSVs to a single Excel workbook.

This small utility collects:
  - ring_vco_fv.csv  -> sheet 'ring_fv'
  - lc_vco_fv.csv    -> sheet 'lc_fv'
  - simulation_summary.csv -> sheet 'summary'

Usage:
  python export_results.py

Output:
  simulation_results.xlsx (in the project root of EE5295_Sim_Scaffold)
"""

from __future__ import annotations

import os
import pandas as pd


def main() -> None:
    base = os.path.join("spice", "results")
    out_xlsx = "simulation_results.xlsx"

    def read_csv_safe(name: str) -> pd.DataFrame:
        path = os.path.join(base, name)
        return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

    ring = read_csv_safe("ring_vco_fv.csv")
    lc = read_csv_safe("lc_vco_fv.csv")
    summary = read_csv_safe("simulation_summary.csv")

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        if not ring.empty:
            ring.to_excel(w, sheet_name="ring_fv", index=False)
        if not lc.empty:
            lc.to_excel(w, sheet_name="lc_fv", index=False)
        if not summary.empty:
            summary.to_excel(w, sheet_name="summary", index=False)
    print(f"Wrote {out_xlsx}")


if __name__ == "__main__":
    main()


