# run_vco_sweep.py
# Use TEMPLATE_vco_ring.cir.j2 + data/inputs_vco.csv to sweep VCTRL and collect FREQ

import os, subprocess, re
import pandas as pd
from jinja2 import Template

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TPL = os.path.join(ROOT,"spice","TEMPLATE_vco_ring.cir.j2")
CSV = os.path.join(ROOT,"data","inputs_vco.csv")
OUT = os.path.join(ROOT,"data","results_vco.csv")
WORK = os.path.join(ROOT,"spice")

MEAS = re.compile(r'^\s*\*?\s*measure\s+\w+\s*:\s*(\w+)\s*=\s*([Ee0-9+\-\.]+)', re.I)

def load_tpl():
    with open(TPL,"r",encoding="utf-8") as f:
        return Template(f.read())

def render(tpl,row,idx):
    params = {k:v for k,v in row.items() if k not in ["title","results_tran_csv"] and v!=""}
    ctx = {
        "title": row.get("title",f"VCO-{idx}"),
        "params": params,
        "VDD": float(row["VDD"]) if "VDD" in row and row["VDD"] else None,
        "VCTRL": float(row["VCTRL"]) if "VCTRL" in row and row["VCTRL"] else None,
        "results_tran_csv": row.get("results_tran_csv", f"vco_run_{idx}.csv"),
        **params,
    }
    out = tpl.render(**ctx)
    path = os.path.join(WORK,f"vco_run_{idx}.cir")
    with open(path,"w",encoding="utf-8") as f:
        f.write(out)
    return path

def run(cir):
    log = cir.replace(".cir",".log")
    cmd = ["ngspice","-b",os.path.basename(cir),"-o",os.path.basename(log)]
    p = subprocess.run(cmd, cwd=WORK, capture_output=True, text=True)
    return os.path.join(WORK, os.path.basename(log))

def parse(log):
    out = {}
    with open(log,"r",encoding="utf-8",errors="ignore") as f:
        for line in f:
            m = MEAS.search(line)
            if m:
                out[m.group(1)] = float(m.group(2))
    return out

def main():
    df = pd.read_csv(CSV)
    tpl = load_tpl()
    rows = []
    for i,row in df.iterrows():
        cir = render(tpl, row.to_dict(), i)
        log = run(cir)
        meas = parse(log)
        rows.append({**row.to_dict(), **meas, "cir": os.path.basename(cir), "log": os.path.basename(log)})
    outdf = pd.DataFrame(rows)
    outdf.to_csv(OUT, index=False)
    print("Wrote", OUT)
    # Quick preview
    try:
        print(outdf[["title","VCTRL","FREQ"]])
    except Exception as e:
        pass

if __name__ == "__main__":
    main()
