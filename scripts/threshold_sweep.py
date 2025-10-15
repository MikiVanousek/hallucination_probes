import argparse, yaml, subprocess, re, os, sys, tempfile

# Strip ANSI color codes from evaluate.py output
ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Matches dataset header: ===== Evaluation Metrics (DATASET NAME) =====
PAT_DATASET = re.compile(r"^=+\s*Evaluation Metrics\s*\((.*?)\)\s*=+", re.IGNORECASE)

# Section & metric lines (tolerant to leading spaces)
PAT_SECTION = re.compile(r"Span\s*Max\s*-\s*Classification\s*Metrics\s*:", re.IGNORECASE)
PAT_METRIC  = re.compile(r"\s*-\s*(Accuracy|Precision|Recall|F1\s*Score)\s*:\s*([0-9.]+)", re.IGNORECASE)

def run_eval(cfg_path):
    p = subprocess.run(
        [sys.executable, "-m", "probe.evaluate", "--config", cfg_path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False
    )
    return ANSI.sub("", p.stdout)  # strip ANSI

def parse_output(text):
    """
    Returns dict: { dataset_name: {acc,prec,rec,f1} } for Span-Max.
    """
    lines = text.splitlines()
    results = {}
    current_ds = None

    i = 0
    while i < len(lines):
        mds = PAT_DATASET.match(lines[i].strip())
        if mds:
            current_ds = mds.group(1).strip()
            # scan forward to Span Max block within this dataset
            j = i + 1
            found = False
            while j < len(lines) and not PAT_DATASET.match(lines[j].strip()):
                if PAT_SECTION.search(lines[j]):
                    # next ~10 lines hold the four metrics
                    acc = prec = rec = f1 = None
                    for k in range(j+1, min(j+12, len(lines))):
                        mm = PAT_METRIC.match(lines[k])
                        if mm:
                            name = mm.group(1).strip().lower().replace(" ", "")
                            val = float(mm.group(2))
                            if name == "accuracy": acc = val
                            elif name == "precision": prec = val
                            elif name == "recall": rec = val
                            elif name == "f1score": f1 = val
                    if acc is not None:
                        results[current_ds] = dict(acc=acc, prec=prec, rec=rec, f1=f1)
                        found = True
                        break
                j += 1
            if not found:
                results[current_ds] = dict(acc=None, prec=None, rec=None, f1=None)
            i = j
        else:
            i += 1
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_cfg", default="configs/meditron_eval_min.yaml")
    ap.add_argument("--thresholds", nargs="+", type=float, required=True)
    ap.add_argument("--dataset_filter", type=str, default=None,
                    help="Only aggregate datasets whose name contains this substring (case-insensitive).")
    args = ap.parse_args()

    base = yaml.safe_load(open(args.base_cfg))
    rows = []  # (thr, macro_acc, macro_prec, macro_rec, macro_f1)
    per_thr_details = {}  # thr -> {ds -> metrics}

    for t in args.thresholds:
        cfg = dict(base)
        cfg["probe_config"] = dict(base["probe_config"])
        cfg["probe_config"]["threshold"] = t
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
            tmp = f.name
        out = run_eval(tmp)
        os.remove(tmp)

        parsed = parse_output(out)
        per_thr_details[t] = parsed

        # filter datasets if requested
        items = list(parsed.items())
        if args.dataset_filter:
            filt = args.dataset_filter.lower()
            items = [(k,v) for k,v in items if filt in k.lower()]

        # macro-average over available datasets
        accs = [v["acc"] for _,v in items if v["acc"] is not None]
        precs = [v["prec"] for _,v in items if v["prec"] is not None]
        recs = [v["rec"] for _,v in items if v["rec"] is not None]
        f1s = [v["f1"] for _,v in items if v["f1"] is not None]

        def mean(xs): return sum(xs)/len(xs) if xs else 0.0
        rows.append((t, mean(accs), mean(precs), mean(recs), mean(f1s)))

    # Print macro table
    print("threshold,spanmax_acc,spanmax_prec,spanmax_rec,spanmax_f1")
    for t,acc,prec,rec,f1 in rows:
        print(f"{t:.4f},{acc:.4f},{prec:.4f},{rec:.4f},{f1:.4f}")

    # Also print per-dataset breakdown for the best F1
    best = max(rows, key=lambda r: r[4] if r[4] is not None else -1)
    tbest = best[0]
    print(f"\n# Best threshold by macro F1: {tbest:.2f}\n# Per-dataset (Span-Max) at {tbest:.2f}:")
    det = per_thr_details[tbest]
    for ds, m in det.items():
        if any(v is None for v in m.values()): continue
        print(f"{ds}: acc={m['acc']:.4f}, prec={m['prec']:.4f}, rec={m['rec']:.4f}, f1={m['f1']:.4f}")

if __name__ == "__main__":
    main()
