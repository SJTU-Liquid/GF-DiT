#!/usr/bin/env bash
# Run the mixed-priority video DiT serving experiment grid:
#   workloads (profiles) x policies, using the new fullrange cost model.
#
# Per-workload, all policies run in a single runner invocation, then a
# Python aggregator pulls every results.json and emits one markdown
# table + one CSV summarizing dataset x policy.
#
# Usage:
#   bash scripts/run-edf-policy-grid.sh
#   bash scripts/run-edf-policy-grid.sh --quick                # short profile only
#   bash scripts/run-edf-policy-grid.sh --policies "static_sp4 edf_greedy"
#   bash scripts/run-edf-policy-grid.sh --out-dir out/grid-v2 --rate-scale 1.5
#
# Output layout:
#   $OUT_DIR/
#     workloads/<profile>.json
#     runs/<profile>.results.json
#     traces/<profile>/<policy>.json    (chrome traces, if --no-traces not set)
#     summary.csv
#     summary.md
set -euo pipefail

OUT_DIR="${OUT_DIR:-out/policy-grid}"
COST_MODEL_DIR="${COST_MODEL_DIR:-cost-model/wan22-ti2v-5b-fullrange}"
RESHARD_MS="${RESHARD_MS:-3.0}"
TOPOLOGY_STYLE="${TOPOLOGY_STYLE:-c84}"
SEED="${SEED:-42}"
DURATION_S=""
RATE_SCALE=""
EMIT_TRACES=1

# Defaults exercise everything the policy file exposes. Drop / extend at will.
PROFILES_DEFAULT="short tridentserve foreground_burst inflight_burst"
POLICIES_DEFAULT="static_sp1 static_sp2 static_sp4 static_sp8 rssp load_adaptive priority_load_adaptive step_preempt_pause step_preempt_demote clean_edf_pause clean_edf_demote edf_greedy edf_best_fit"

PROFILES="$PROFILES_DEFAULT"
POLICIES="$POLICIES_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)            PROFILES="short"; shift;;
    --out-dir)          OUT_DIR="$2"; shift 2;;
    --cost-model-dir)   COST_MODEL_DIR="$2"; shift 2;;
    --profiles)         PROFILES="$2"; shift 2;;
    --policies)         POLICIES="$2"; shift 2;;
    --reshard-ms)       RESHARD_MS="$2"; shift 2;;
    --topology-style)   TOPOLOGY_STYLE="$2"; shift 2;;
    --duration-s)       DURATION_S="$2"; shift 2;;
    --rate-scale)       RATE_SCALE="$2"; shift 2;;
    --seed)             SEED="$2"; shift 2;;
    --no-traces)        EMIT_TRACES=0; shift 1;;
    -h|--help)
      sed -n '1,30p' "$0" | sed 's/^# \{0,1\}//'
      exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Resolve python: prefer .venv if present, else system.
PY="python3"
[[ -x "$REPO_ROOT/.venv/bin/python" ]] && PY="$REPO_ROOT/.venv/bin/python"

WORKLOAD_DIR="$OUT_DIR/workloads"
RUNS_DIR="$OUT_DIR/runs"
TRACE_DIR="$OUT_DIR/traces"
mkdir -p "$WORKLOAD_DIR" "$RUNS_DIR"

# Sanity-check the SP=1 cost-model file the workload generator needs.
SP1_FILE="$COST_MODEL_DIR/tp1_ulysses1_ring1_cfg1.json"
if [[ ! -f "$SP1_FILE" ]]; then
  echo "missing SP=1 cost-model file at $SP1_FILE" >&2
  echo "run scripts/profile-stages-fullrange.sh first, or pass --cost-model-dir" >&2
  exit 1
fi

# Detect which SP sizes the cost-model dir actually covers. Any policy that
# touches an unfit SP (e.g. static_sp8 with no SP=8 cost-model) errors at
# dispatch and the result row would be blank. Cap the topology + edf_greedy
# allowed sizes to detected SPs so the experiment grid stays coherent.
AVAILABLE_SPS=$(ls "$COST_MODEL_DIR"/tp1_ulysses*_ring1_cfg1.json 2>/dev/null \
  | sed -E 's/.*ulysses([0-9]+).*/\1/' \
  | sort -n | uniq | tr '\n' ' ')
if [[ -z "$AVAILABLE_SPS" ]]; then
  echo "no cost-model files (tp1_ulysses*_ring1_cfg1.json) found in $COST_MODEL_DIR" >&2
  exit 1
fi
echo "detected cost-model SPs: $AVAILABLE_SPS"

# Generate a custom topology JSON that only includes SPs we have cost-models
# for. Uses the c84 enumeration shape (every C(8,k) rank subset at each SP)
# so all non-edf_greedy policies still have multiple SP-k lanes to pick from.
TOPOLOGY_FILE="$OUT_DIR/topology.json"
mkdir -p "$OUT_DIR"
"$PY" - "$TOPOLOGY_FILE" $AVAILABLE_SPS <<'PYEOF'
import itertools, json, sys

out_path = sys.argv[1]
sps = sorted({int(s) for s in sys.argv[2:]})
WORKERS = 8
groups = []
if 1 in sps:
    for i in range(WORKERS):
        groups.append({
            "group_id": f"sp1_{i}", "ranks": [i],
            "tp": 1, "ulysses_degree": 1, "ring_degree": 1, "cfg": 1,
        })
for sp in sps:
    if sp == 1 or sp > WORKERS:
        continue
    for idx, ranks in enumerate(itertools.combinations(range(WORKERS), sp)):
        groups.append({
            "group_id": f"sp{sp}_{idx}", "ranks": list(ranks),
            "tp": 1, "ulysses_degree": sp, "ring_degree": 1, "cfg": 1,
        })
with open(out_path, "w") as fh:
    json.dump(groups, fh)
print(f"wrote {out_path} with {len(groups)} groups for SP in {sps}")
PYEOF

# Auto-set EDF_GREEDY_SP_SIZES if the caller didn't.
if [[ -z "${EDF_GREEDY_SP_SIZES:-}" ]]; then
  export EDF_GREEDY_SP_SIZES=$(echo "$AVAILABLE_SPS" | tr ' ' ',' | sed 's/,$//')
  echo "auto-set EDF_GREEDY_SP_SIZES=$EDF_GREEDY_SP_SIZES"
else
  echo "using caller's EDF_GREEDY_SP_SIZES=$EDF_GREEDY_SP_SIZES"
fi

# Build policy spec args: --policy module:make_X module:make_Y ... --policy-name X Y ...
POLICY_MODULE="benchmarks.diffusion.runtime_v2_mixed_priority_policies"
POLICY_SPECS=()
POLICY_NAMES=()
for name in $POLICIES; do
  # Drop static_spN entries when SP=N isn't in the cost-model.
  if [[ "$name" =~ ^static_sp([0-9]+)$ ]]; then
    sp="${BASH_REMATCH[1]}"
    if ! echo " $AVAILABLE_SPS " | grep -q " $sp "; then
      echo "skip $name (cost-model has no SP=$sp)"
      continue
    fi
  fi
  POLICY_SPECS+=("$POLICY_MODULE:make_$name")
  POLICY_NAMES+=("$name")
done
if [[ ${#POLICY_SPECS[@]} -eq 0 ]]; then
  echo "no runnable policies after filtering against cost-model SPs" >&2
  exit 1
fi

echo "out_dir=$OUT_DIR"
echo "cost_model=$COST_MODEL_DIR"
echo "topology_file=$TOPOLOGY_FILE  reshard_ms=$RESHARD_MS"
echo "profiles: $PROFILES"
echo "policies: ${POLICY_NAMES[*]}"
echo

for profile in $PROFILES; do
  workload="$WORKLOAD_DIR/$profile.json"
  if [[ ! -f "$workload" ]]; then
    echo "[gen] $profile -> $workload"
    gen_args=(
      --out "$workload"
      --cost-model-dir "$COST_MODEL_DIR"
      --profile "$profile"
      --seed "$SEED"
    )
    [[ -n "$DURATION_S" ]] && gen_args+=(--duration-s "$DURATION_S")
    [[ -n "$RATE_SCALE" ]] && gen_args+=(--rate-scale "$RATE_SCALE")
    "$PY" benchmarks/diffusion/generate_runtime_v2_mixed_priority_workload.py "${gen_args[@]}"
  else
    echo "[gen] $profile reusing $workload"
  fi

  results_out="$RUNS_DIR/$profile.results.json"
  echo "[run] $profile ($(echo $POLICIES | wc -w) policies)"
  trace_args=()
  if [[ "$EMIT_TRACES" -eq 1 ]]; then
    profile_trace="$TRACE_DIR/$profile"
    mkdir -p "$profile_trace"
    trace_args=(--trace-out-dir "$profile_trace")
  fi
  PYTHONPATH=. "$PY" benchmarks/diffusion/runtime_v2_mixed_priority_runner.py \
      --workload-json "$workload" \
      --cost-model-dir "$COST_MODEL_DIR" \
      --policy "${POLICY_SPECS[@]}" \
      --policy-name "${POLICY_NAMES[@]}" \
      --topology-json "$(cat "$TOPOLOGY_FILE")" \
      --reshard-ms "$RESHARD_MS" \
      --results-out "$results_out" \
      "${trace_args[@]}"
  echo
done

# -----------------------------------------------------------------------------
# Aggregator: read every runs/*.results.json, emit summary.{csv,md}.
# -----------------------------------------------------------------------------
"$PY" - "$RUNS_DIR" "$OUT_DIR/summary.csv" "$OUT_DIR/summary.md" <<'PYEOF'
import csv
import json
import sys
from pathlib import Path

runs_dir = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
md_path = Path(sys.argv[3])

# Columns we want per (dataset, policy):
#   makespan_s, idle_frac, S/M/L slo_attainment, S/M/L p95 lat (s),
#   S/M/L p95 normalized.
KEY_COLS = ("dataset", "policy")
NUM_COLS = (
    "makespan_s",
    "idle_frac",
    "S_slo",  "S_p95_s",  "S_n95",
    "M_slo",  "M_p95_s",  "M_n95",
    "L_slo",  "L_p95_s",  "L_n95",
)
ALL_COLS = KEY_COLS + NUM_COLS


def _ms_to_s(value):
    if value is None:
        return None
    return float(value) / 1000.0


def _class_view(by_class, klass):
    info = by_class.get(klass) or {}
    return (
        info.get("slo_attainment_ratio"),
        _ms_to_s(info.get("p95_latency_ms")),
        info.get("p95_normalized_latency"),
    )


rows = []
errors: list[tuple[str, str, str]] = []  # (dataset, policy, error)
for results_path in sorted(runs_dir.glob("*.results.json")):
    dataset = results_path.stem.removesuffix(".results")
    data = json.loads(results_path.read_text())
    for run in data.get("results", []):
        policy = run.get("name") or run.get("policy")
        if "error" in run:
            errors.append((dataset, policy, run["error"]))
            rows.append({
                "dataset": dataset, "policy": policy,
                "makespan_s": None, "idle_frac": None,
                "S_slo": None, "S_p95_s": None, "S_n95": None,
                "M_slo": None, "M_p95_s": None, "M_n95": None,
                "L_slo": None, "L_p95_s": None, "L_n95": None,
                "_failed": True,
            })
            continue
        overall = run.get("overall", {}) or {}
        by_class = run.get("by_class", {}) or {}
        s_slo, s_p95, s_n95 = _class_view(by_class, "S")
        m_slo, m_p95, m_n95 = _class_view(by_class, "M")
        l_slo, l_p95, l_n95 = _class_view(by_class, "L")
        rows.append({
            "dataset": dataset,
            "policy": policy,
            "makespan_s": _ms_to_s(overall.get("makespan_ms")),
            "idle_frac": overall.get("global_idle_frac"),
            "S_slo": s_slo, "S_p95_s": s_p95, "S_n95": s_n95,
            "M_slo": m_slo, "M_p95_s": m_p95, "M_n95": m_n95,
            "L_slo": l_slo, "L_p95_s": l_p95, "L_n95": l_n95,
        })

if not rows:
    print("no results found under", runs_dir, file=sys.stderr)
    sys.exit(1)

# CSV (raw, full precision).
with csv_path.open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=ALL_COLS)
    writer.writeheader()
    for row in rows:
        writer.writerow({col: row[col] for col in ALL_COLS})

# Markdown (compact, 3 sig figs, with policy as row and dataset-stat as col).
def fmt(val, kind):
    if val is None:
        return "—"
    if kind == "slo":
        return f"{val * 100:.0f}%"
    if kind == "frac":
        return f"{val * 100:.1f}%"
    if kind == "n95":
        return f"{val:.2f}x"
    return f"{val:.1f}"


# Group rows by dataset for a wide table.
datasets = sorted({row["dataset"] for row in rows})
policies = list(dict.fromkeys(row["policy"] for row in rows))  # preserve run order


failed_pairs = {(d, p) for d, p, _ in errors}


def render_section(metric_label, getter, fmt_kind):
    header = "| policy | " + " | ".join(datasets) + " |"
    sep    = "|" + "---|" * (len(datasets) + 1)
    lines  = [f"### {metric_label}", "", header, sep]
    by_pair = {(r["dataset"], r["policy"]): getter(r) for r in rows}
    for policy in policies:
        cells = []
        for d in datasets:
            if (d, policy) in failed_pairs:
                cells.append("ERR")
            else:
                cells.append(fmt(by_pair.get((d, policy)), fmt_kind))
        lines.append("| " + policy + " | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


md_chunks = [
    "# Mixed-priority policy x dataset summary",
    "",
    f"datasets: {', '.join(datasets)}",
    f"policies: {', '.join(policies)}",
    "",
]
if errors:
    md_chunks.append("### Failed runs (ERR cells)")
    md_chunks.append("")
    for dataset, policy, err in errors:
        md_chunks.append(f"- **{dataset} / {policy}**: `{err}`")
    md_chunks.append("")
md_chunks += [
    render_section("Makespan (s)",          lambda r: r["makespan_s"],   "raw"),
    render_section("Global idle (%)",       lambda r: r["idle_frac"],    "frac"),
    render_section("S — SLO attainment",    lambda r: r["S_slo"],        "slo"),
    render_section("S — p95 latency (s)",   lambda r: r["S_p95_s"],      "raw"),
    render_section("S — p95 normalized",    lambda r: r["S_n95"],        "n95"),
    render_section("M — SLO attainment",    lambda r: r["M_slo"],        "slo"),
    render_section("M — p95 latency (s)",   lambda r: r["M_p95_s"],      "raw"),
    render_section("M — p95 normalized",    lambda r: r["M_n95"],        "n95"),
    render_section("L — SLO attainment",    lambda r: r["L_slo"],        "slo"),
    render_section("L — p95 latency (s)",   lambda r: r["L_p95_s"],      "raw"),
    render_section("L — p95 normalized",    lambda r: r["L_n95"],        "n95"),
]
md_path.write_text("\n".join(md_chunks))

print(f"\nwrote {csv_path}")
print(f"wrote {md_path}")
print()
# Echo markdown to stdout for quick eyeballing.
print(md_path.read_text())
PYEOF
