#!/usr/bin/env bash
# Unified per-device overflow gate ladder (issue #30).
#
# One ladder per device, in covenant order; the first stage that
# fails stops the war (STAGE FAILED). No gate logic lives here -
# this script only chains the gates as the covenant prescribes:
# live capture -> coverage audit -> record -> noise -> static
# capture gate -> replay calibration -> overflow verdict.
#
# Usage:
#   scripts/run_overflow_gate.sh LEAK_SLOPE_MB MIN_AVAILABLE_GB PER_SHARD_GB
#
#   LEAK_SLOPE_MB     measured slope of the 256-token leak probe, MB/token
#   MIN_AVAILABLE_GB  smallest measured free memory across devices, GB
#   PER_SHARD_GB      largest per-shard memory footprint observed in the run, GB
#
# All three are MEASURED facts; none may be guessed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

H="${HOSTNAME:-$(hostname 2>/dev/null || echo device)}"
PY="${PYTHON:-python3}"

if [ "$#" -ne 3 ]; then
    echo "usage: $0 LEAK_SLOPE_MB MIN_AVAILABLE_GB PER_SHARD_GB" >&2
    echo "  all three are measured facts from the run; none may be guessed" >&2
    exit 2
fi
LEAK_SLOPE_MB="$1"
MIN_AVAILABLE_GB="$2"
PER_SHARD_GB="$3"

fail() {
    echo "STAGE FAILED: $1"
    exit 1
}

run_stage() {
    local stage="$1"
    shift
    echo "=== STAGE: ${stage} ==="
    if ! "$@"; then
        fail "${stage}"
    fi
}

# the measured model facts flow from the ledger, never from paper
MODEL_GB="$(awk -F': *' '/^total_size_gb:/ {print $2}' models/overflow.yaml)"
[ -n "$MODEL_GB" ] || fail "model facts (total_size_gb missing in models/overflow.yaml)"

# 1. live capture gate - the recorder judged while it records
run_stage "live-capture-gate" \
    "$PY" benchmark.py --live-capture-gate --report "live_${H}.json"

# 2. coverage audit - the journal matched against reality
run_stage "coverage-audit" \
    "$PY" benchmark.py --coverage-audit --report "audit_${H}.json"

# 3. record the workload trace (the live gate flag is carried and
# the audit executes inside --record-trace, per the ladder)
run_stage "record-trace" \
    "$PY" benchmark.py --live-capture-gate --record-trace \
    --model-gb "$MODEL_GB" --tokens 16 \
    --trace-path "trace_${H}.json" --report "trace_${H}.json"

# 4. noise floor - repeat until POWERED (amendment #25: the ruler
# is also measured); a hard error fails immediately, an
# under-powered report grows N by 10 up to the declared cap
N=10
while :; do
    echo "=== STAGE: noise (N=${N}) ==="
    if ! "$PY" benchmark.py --stage 1 --variance-report "$N" --paired \
            --report "noise_${H}.json"; then
        fail "noise"
    fi
    STATUS="$("$PY" -c "import json; print('POWERED' if not json.load(open('noise_${H}.json'))['underpowered'] else 'UNDERPOWERED')")"
    [ "$STATUS" = "POWERED" ] && break
    N=$((N + 10))
    [ "$N" -le 200 ] || fail "noise (cap of 200 repeats reached without POWERED)"
done

# 5. static capture gate (the audit flag is carried and the live
# gate executes inside it, per the ladder)
run_stage "trace-capture-gate" \
    "$PY" benchmark.py --coverage-audit --trace-capture-gate \
    --trace-path "trace_${H}.json" --report "capgate_${H}.json"

# 6. replay calibration (inherits the whole chain through
# --trace-capture-gate)
run_stage "replay-calibrate" \
    "$PY" benchmark.py --replay-calibrate --trace-capture-gate \
    --coverage-audit --trace-path "trace_${H}.json" \
    --noise-report "noise_${H}.json" --report "calib_${H}.json"

# 7. overflow verdict - the tool is the judge, the runner supplies
# only measured facts; the calibration numbers leave
# calib_${H}.json, never a human memory
MAX_USABLE="$("$PY" -c "import json; print(json.load(open('calib_${H}.json'))['max_usable_shard_gb'])")" \
    || fail "calibration facts (max_usable_shard_gb missing in calib_${H}.json)"
FLAT="$("$PY" -c "import json; print(json.load(open('calib_${H}.json'))['flat_ceiling_gb'])")" \
    || fail "calibration facts (flat_ceiling_gb missing in calib_${H}.json)"
run_stage "overflow-verdict" \
    "$PY" benchmark.py --gate overflow --model-gb "$MODEL_GB" --shards 3 \
    --per-shard-gb "$PER_SHARD_GB" --completed-tokens 64 \
    --min-available-gb "$MIN_AVAILABLE_GB" \
    --max-usable-shard-gb "$MAX_USABLE" \
    --leak-slope-mb "$LEAK_SLOPE_MB" --flat-shard-gb "$FLAT" \
    --report "verdict_${H}.json"

echo "LADDER COMPLETE: verdict_${H}.json"
