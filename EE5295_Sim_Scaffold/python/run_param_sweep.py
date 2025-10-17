# ---------- run_param_sweep.py ----------
# Usage:
#   cd python
#   python run_param_sweep.py
#
# It reads ../data/inputs.csv, fills spice/TEMPLATE_netlist.cir.j2
# for each row, runs ngspice in batch, parses `.measure` from log,
# and appends results to ../data/results.csv

import os, csv, subprocess, re, sys
from jinja2 import Template
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPICE_TPL = os.path.join(ROOT, "spice", "TEMPLATE_netlist.cir.j2")
DATA_IN   = os.path.join(ROOT, "data", "inputs.csv")
DATA_OUT  = os.path.join(ROOT, "data", "results.csv")
WORKDIR   = os.path.join(ROOT, "spice")

MEASURE_RE = re.compile(r"^\\s*\\*?\\s*measure\\s+\\w+\\s*:\\s*(\\w+)\\s*=\\s*([eE0-9+\\-.]+)", re.IGNORECASE)

def load_template(path):
    with open(path, "r", encoding="utf-8") as f:
        return Template(f.read())

def render_netlist(tpl, row, idx):
    # Standard keys you can pass via CSV:
    # title, VDD, results_ac_csv, any custom params (R1,C1,...)
    params = {k: v for k, v in row.items() if k not in ["title", "results_ac_csv"] and v != ""}
    ctx = {
        "title": row.get("title", f"Run {idx}"),
        "params": params,
        "VDD": float(row["VDD"]) if "VDD" in row and row["VDD"] else None,
        "results_ac_csv": row.get("results_ac_csv", f"ac_run_{idx}.csv"),
        **{k: row[k] for k in params.keys()},
    }
    content = tpl.render(**ctx)
    outpath = os.path.join(WORKDIR, f"generated_run_{idx}.cir")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(content)
    return outpath

def run_ngspice(cir_path):
    log_path = cir_path.replace(".cir", ".log")
    cmd = ["ngspice", "-b", os.path.basename(cir_path), "-o", os.path.basename(log_path)]
    proc = subprocess.run(cmd, cwd=WORKDIR, capture_output=True, text=True)
    if proc.returncode != 0:
        print("NGSpice failed:", proc.stderr)
    return os.path.join(WORKDIR, os.path.basename(log_path))

def parse_measures(log_path):
    results = {}
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = MEASURE_RE.search(line)
            if m:
                name, val = m.groups()
                try:
                    results[name] = float(val)
                except:
                    results[name] = val
    return results

def main():
    tpl = load_template(SPICE_TPL)
    df = pd.read_csv(DATA_IN)
    rows = []
    for i, row in df.iterrows():
        cir = render_netlist(tpl, row.to_dict(), i)
        log = run_ngspice(cir)
        meas = parse_measures(log)
        rows.append({**row.to_dict(), **meas, "cir": os.path.basename(cir), "log": os.path.basename(log)})
    out_df = pd.DataFrame(rows)
    if os.path.exists(DATA_OUT):
        prev = pd.read_csv(DATA_OUT)
        out_df = pd.concat([prev, out_df], ignore_index=True)
    out_df.to_csv(DATA_OUT, index=False)
    print(f"Wrote {DATA_OUT} with {len(rows)} rows.")

if __name__ == "__main__":
    main()
