#!/usr/bin/env python3
"""Generate a hot-expert deployment config from a hotstats JSON snapshot.

The output config is the deployment manifest: which (layer, expert) pairs to
place on which device. You can edit it by hand - remove experts when VRAM is
tight, add more when you have headroom. A later surgery step consumes this file.

Usage:
  llama_hotcfg.py hot_experts.json [--layers N..42] [--top N] [--target-float-mb N]
                 [--dev CUDA0] [--out cfg.json] [--print]

Config format (JSON):
  {
    "devices": { "CUDA0": { "4": [101, 50, 71], "8": [200, 189] } },
    "total_experts": 860,
    "est_vram_mb": 10062,
    "per_expert_mb": 11.7
  }

Selection policy per layer: take the top-N experts from the snapshot's
per-layer top lists. VRAM pinning honors --target-float-mb and --top whichever
is tighter.
"""

import argparse, json, sys


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("snapshot", help="hotstats snapshot json (hot_experts.json)")
    p.add_argument("--layers", default="3-42",
                   help="layer range to select from, e.g. 3-42 (default)")
    p.add_argument("--top", type=int, default=16,
                   help="experts per layer to take from the top list (default 16)")
    p.add_argument("--target-vram-mb", type=int, default=10000,
                   help="VRAM budget in MB; shrink expert sets to fit (default 10000)")
    p.add_argument("--dev", default="CUDA0", help="device to place hot experts on")
    p.add_argument("--out", default="hot_cfg.json", help="output config file")
    p.add_argument("--print", action="store_true", help="print stats to stdout")
    p.add_argument("--emit-param", action="store_true",
                   help="after writing cfg, print compact llama.cpp param string")
    p.add_argument("--load", metavar="CFG.JSON",
                   help="load an existing (hand-edited) cfg instead of generating; "
                        "then only --emit-param applies")
    return p.parse_args()


def parse_layer_range(spec):
    """Parse '3-42' or '3,5,7-9' into a set of ints."""
    ids = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
            ids.update(range(a, b + 1))
        else:
            ids.add(int(part))
    return ids


def main():
    args = parse_args()

    with open(args.snapshot) as f:
        snap = json.load(f)

    layers = snap.get("layers", {})
    wanted = parse_layer_range(args.layers)

    # per-layer top lists, most frequent first
    per_layer = {}
    for key, lc in layers.items():
        try:
            il = int(key.split(".")[1])
        except (IndexError, ValueError):
            continue
        if il not in wanted:
            continue
        ranked = sorted(lc.get("top", []), key=lambda e: -e["c"])
        per_layer[il] = ranked

    if not per_layer:
        sys.exit("no layer data found in snapshot for range %s" % args.layers)

    # estimate per-expert VRAM from snapshot (layers blk.0 size hint unused here;
    # use a configurable global constant, printed for reference)
    per_expert_mb = 11.7  # IQ4_XS 256 experts / exps tensor ~ 2.99GB; change per model

    # pick per layer: up to --top experts, then trim uniformly to fit VRAM
    def trim_to_budget(sets, budget_mb):
        total_n = sum(len(v) for v in sets.values())
        while total_n * per_expert_mb > budget_mb:
            # drop the least frequent expert of the largest layer set
            best = max(sets, key=lambda l: len(sets[l]))
            sets[best].pop()  # list is already sorted by frequency desc
            total_n -= 1
            if total_n == 0:
                break
        return total_n

    picked = {il: [e["e"] for e in per_layer[il][: args.top]] for il in per_layer}
    total = trim_to_budget(picked, args.target_vram_mb)

    cfg = {
        "devices": {args.dev: {str(il): picked[il] for il in sorted(picked) if picked[il]}},
        "total_experts": total,
        "est_vram_mb": int(total * per_expert_mb),
        "per_expert_mb": per_expert_mb,
    }

    with open(args.out, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")

    if args.print:
        print("wrote %s" % args.out)
        print("layers: %d (%s)" % (len(picked), args.layers))
        print("experts total: %d, est VRAM: %d MB (%.1f MB each)"
              % (total, cfg["est_vram_mb"], per_expert_mb))
        dev = args.dev
        print("config: device=%s" % dev)
        for il in sorted(picked):
            if picked[il]:
                print("  %d:%s" % (il, ",".join(str(e) for e in picked[il])))


if __name__ == "__main__":
    main()