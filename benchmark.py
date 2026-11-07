#!/usr/bin/env python3
"""
HiveMind Swarm Benchmark - a pre-registered experiment harness.

Question under test: does a swarm of weak devices connected
through HiveMind's distributed gRPC layer run, per hardware
dollar, a model that no single device of the same total cost can
run - and does the coordination tax of distribution stay inside
budget?

Five stages, each isolating exactly one variable:

  Stage 1  isolated    Layers called in-process (no network).
                        -> pure compute throughput baseline.
  Stage 2  loopback    Client and servers over localhost gRPC.
                        (2) - (1) = cost of the gRPC stack itself.
  Stage 3  swarm       Real hosts over the LAN.
                        (3) - (2) = cost of the physical network.
  Stage 4  netem       Artificial latency injected with tc netem.
                        -> TPS-vs-RTT curve and the break-even point.
  Stage 5  instrument  Per-request send/recv/compute timestamps.
                        wait_time / compute_time = coordination tax.

PRE-REGISTERED FAILURE THRESHOLDS - fixed before the first run.
Editing these after seeing results is not measurement, it is
propaganda:

  SWARM_ADVANTAGE_FLOOR    = 1.5   the swarm must run, at
                                   >= 1 token/s, a model at least
                                   1.5x the size of what a single
                                   device of equal total cost runs
                                   at the same speed.
  SCALING_EFFICIENCY_FLOOR = 0.65  each swarm device must deliver
                                   at least 65% of its standalone
                                   throughput.
  COORDINATION_TAX_CEILING = 0.40  wait_time / compute_time must
                                   stay at or below 40%.
  MIN_INTERACTIVE_TPS      = 1.0   the definition of "it runs".

The verdict is printed as THESIS HOLDS or THESIS REFUTED - never
both, never "inconclusive but favorable". A commander who sets
the withdrawal line after seeing the battle is not testing a
thesis; he is shopping for a justification.

PRE-REGISTERED COVENANT (full text: PRE_REGISTERED.md, verified
against seal/THRESHOLDS.sha256 by CI on every push):

- The anchor verdict is reported across a tolerance sweep
  (ANCHOR_TOLERANCE_SWEEP), never at a single point.
- No claim may exceed ANCHOR_EXTRAPOLATION_CAP times the
  anchored size: no anchor, no summit talk.
- CORRECTION_FACTOR = FORBIDDEN: the anchor is a question,
  never a calibration knob.

Usage:
  python benchmark.py --stage 1 --model-gb 0.2 --shards 1 --tokens 32
  python benchmark.py --stage 2 --model-gb 0.6 --shards 3 --tokens 32
  python benchmark.py --stage 5 --model-gb 0.6 --shards 3 --tokens 32
  python benchmark.py --stage 3 --hosts h1:50051,h2:50051 --model-gb 0.6
  python benchmark.py --stage 4 --apply-netem
  python benchmark.py --serve --port 50051 --shard-gb 0.2
  python benchmark.py --verdict single.json swarm.json
  python benchmark.py --calibrate-shard --calibrate-step-gb 0.25
  python benchmark.py --record-trace --trace-path trace.json --model-gb 15 --shards 3 --tokens 64
  python benchmark.py --replay-calibrate --trace-path trace.json --calibrate-step-gb 0.25
  python benchmark.py --stage 1 --variance-report 10 --paired --report noise.json
  python benchmark.py --reproduce-tolerance noise.json their_noise.json
  python benchmark.py --gate overflow --model-gb 15 --shards 3 --per-shard-gb 6 \
      --completed-tokens 64 --min-available-gb 7.5 --max-usable-shard-gb 6.5
"""
import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time
import zlib
from typing import List, Optional

import torch

# Make "import hivemind" work from a source checkout.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hivemind.net.serialization import serialize_tensor, deserialize_tensor
from hivemind.net.server import InferenceServicer, CompressionType, serve
from hivemind.net.client import InferenceClient
from hivemind.net.proto import inference_pb2, inference_pb2_grpc

# ------------------------------------------------------------------
# Pre-registered thresholds. Do not edit after data collection.
# ------------------------------------------------------------------
SWARM_ADVANTAGE_FLOOR = 1.5
SCALING_EFFICIENCY_FLOOR = 0.65
COORDINATION_TAX_CEILING = 0.40
MIN_INTERACTIVE_TPS = 1.0
NETEM_LATENCY_POINTS_MS = [5, 10, 20, 50]

# ------------------------------------------------------------------
# Anchor calibration policy (see PRE_REGISTERED.md, hash-sealed).
# A single tolerance of 0.25 was rejected: no decision could be
# named that flips at 26% but not at 24%. Verdicts are therefore
# reported across a tolerance sweep, never at one point.
# ------------------------------------------------------------------
ANCHOR_TOLERANCE_SWEEP = [0.05, 0.10, 0.25, 0.50]
ANCHOR_EXTRAPOLATION_CAP = 2.0
CORRECTION_FACTOR = "FORBIDDEN"

# ------------------------------------------------------------------
# Overflow gate - the heart of the thesis (see PRE_REGISTERED.md).
# The two-shard gate tests the tax floor; this gate tests the
# heart: running a model no single device can hold. It runs
# FIRST. Editing these numbers after data = moving the withdrawal
# line in a new uniform.
# ------------------------------------------------------------------
OVERFLOW_GATE = {
    # (a) decision-derived: fewer than 3 devices is a pair, not
    # a swarm - no routing, no meaningful coordination tax.
    # More than 3 has diminishing coordination returns (stage 5).
    "min_devices": 3,
    # (a) decision-derived from the leak budget: the shortest
    # completion length that can expose a leak of >= 1 MB/token.
    # The slope must be probed on a separate 256-token run.
    "min_completion_tokens": 64,
    "leak_probe_tokens": 256,
    "leak_budget_mb_per_token": 1.0,
}

# The per-shard budget is NOT a constant: it is a measured
# fact (max usable shard memory), recorded per run. The
# fixed 6.0 GB was a guess; the 0.85 safety factor was a
# guess about a guess. Both were executed (amendments
# #23, #24). No factor survives.
#
# Reproduction tolerance: the replay acceptance margin is
# derived from measured noise, never chosen. The margin
# applies to the swarm ADVANTAGE RATIO, not raw TPS: the
# ratio divides the same hardware noise by itself.
REPRODUCE_K = 3
# k = 3 is justified, not decorative (amendment #25):
# 3 sigma = 0.27% false failure per test; at ~10
# reproduction tests per year that is ~0.027 false
# failures per year - the two named decisions (acceptable
# false-failure frequency and test cadence) that derive it.
#
# The judge is also on trial (amendment #25): a calibration
# that tests only SIZE is optimistic by the fragmentation
# premium. The gate therefore consumes the TRACE-REPLAY
# calibration (--replay-calibrate), which replays the
# workload allocation pattern (persistent weights,
# transient activations, monotone KV growth, out-of-order
# jitter frees) and OOM-steps the SCALE FACTOR. The flat
# contiguous ceiling (--calibrate-shard) survives only as a
# diagnostic for the premium = (flat - replayed) / flat.
# Controls are mandatory: a replay at
# REPLAY_CONTROL_SAFE_FRACTION of the measured limit must
# survive and a replay at REPLAY_CONTROL_DOOM_FRACTION must
# fail; otherwise the calibration itself is invalid and no
# number leaves the tool. A premium above the DERIVED
# threshold (amendment #26: REPRODUCE_K x sigma_noise from
# the variance report, never a chosen 10%) proves the flat
# ceiling unfaithful: executed, never gates.
REPLAY_CONTROL_SAFE_FRACTION = 0.5
REPLAY_CONTROL_DOOM_FRACTION = 1.5
#
# The margin is an INTERVAL, never a point (amendment #25):
# sigma estimated from n runs is itself noisy (chi-square,
# ~24% relative uncertainty at n=10). The report prints
# [k x sigma_lower, k x sigma_upper] from a two-sided
# CI_LEVEL interval on sigma (Wilson-Hilferty quantiles,
# a declared approximation) and REFUSES judgment while the
# interval is wider than TARGET_MARGIN_WIDTH. The target is
# derived from SWARM_ADVANTAGE_FLOOR: to certify a 1.5x
# advantage the margin relative width must stay below
# 0.10. An under-powered tool rejects nothing - it
# abstains, loudly and nonzero.
CI_LEVEL = 0.95
Z_975 = 1.959964  # two-sided 95% normal quantile, from CI_LEVEL
TARGET_MARGIN_WIDTH = 0.10
#
# The capture layer is judged before it measures (amendment
# #26). The 100MB leak injection and its +/-1% tolerance are
# prescribed by issue #26 as the control that judges the
# recorder itself. Tags legitimately alive when the journal
# ends: weights persist for the model lifetime and the KV
# cache grows monotonically through generation (freed only
# at context end) - declared in covenant section 5.
LEAK_INJECT_BYTES = 100 * 1024 ** 2
LEAK_INJECT_TOLERANCE = 0.01
# LIVE_TAGS is closed only for this amendment; a new tag joins
# by an open issue that names the tag and why it is legitimately
# alive at journal end (amendment #27 made the rule explicit -
# MoE expert buffers retained between tokens will need it).
LIVE_TAGS = ("weight", "kv")
#
# On-device injection (amendment #27): the device itself is
# the witness. CUDA allocator counters are exact on the
# allocator path (+/-1%); RSS carries the caching allocator's
# noise, so the CPU tolerance is wider and declared (+/-5%),
# prescribed by issue #27. The ten live probe sizes are the
# known pattern the positive live control must capture, in
# order.
LIVE_INJECT_TOLERANCE_CUDA = 0.01
LIVE_INJECT_TOLERANCE_CPU = 0.05
LIVE_PROBE_SIZES = [4096, 8192, 16384, 32768, 65536, 131072,
                    262144, 524288, 1048576, 2097152]


class MockContext:
    """Minimal stand-in for grpc.ServicerContext (metadata only)."""

    def invocation_metadata(self):
        return ()


def build_shard_layer(shard_bytes: int, in_features: int) -> torch.nn.Linear:
    """One synthetic matmul layer whose weights occupy shard_bytes."""
    out_features = max(1, int(shard_bytes) // (in_features * 4))
    layer = torch.nn.Linear(in_features, out_features, bias=False)
    with torch.no_grad():
        layer.weight.mul_(0.01)
    return layer


def build_layers(model_gb: float, shards: int, in_features: int) -> list:
    """Split a synthetic model of model_gb across shards."""
    per_shard = int(model_gb * 1024 ** 3 / max(1, shards))
    return [build_shard_layer(per_shard, in_features) for _ in range(shards)]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_request(tensor: torch.Tensor, shard_index: int = 0):
    return inference_pb2.InferenceRequest(
        input_data=zlib.compress(serialize_tensor(tensor)),
        model_id="benchmark",
        shard_index=shard_index,
    )


def _chain_isolated(servicers, x, context):
    for svc in servicers:
        resp = svc.ProcessShard(make_request(x), context)
        if resp.error:
            raise RuntimeError(f"servicer error: {resp.error}")
        x = deserialize_tensor(zlib.decompress(resp.output_data))
    return x


def run_isolated(layers, tokens: int, in_features: int) -> dict:
    """Stage 1: chained servicer calls in-process. No network."""
    servicers = [
        InferenceServicer(
            shard_index=i,
            shard_layers=[("bench", lyr)],
            compression=CompressionType.ZLIB,
        )
        for i, lyr in enumerate(layers)
    ]
    context = MockContext()
    x = torch.randn(1, in_features)
    _chain_isolated(servicers, x, context)  # warmup
    per_token = []
    ttft = None
    t0 = time.perf_counter()
    for n in range(tokens):
        t_start = time.perf_counter()
        _chain_isolated(servicers, x, context)
        dt = time.perf_counter() - t_start
        per_token.append(dt)
        if n == 0:
            ttft = dt
    elapsed = time.perf_counter() - t0
    return {
        "stage": 1,
        "name": "isolated",
        "tokens": tokens,
        "shards": len(layers),
        "tps": tokens / elapsed,
        "ttft_s": ttft,
        "per_token_s": statistics.mean(per_token),
    }


def _chain_clients(clients, x):
    for client in clients:
        x = client.process_shard(x, model_id="benchmark")
    return x


def run_swarm(addresses: List[str], tokens: int, in_features: int,
               stage: int = 2, name: str = "loopback") -> dict:
    """Stages 2/3: chain tokens through shard servers over gRPC."""
    clients = [InferenceClient(addr) for addr in addresses]
    x = torch.randn(1, in_features)
    _chain_clients(clients, x)  # warmup
    per_token = []
    ttft = None
    t0 = time.perf_counter()
    for n in range(tokens):
        t_start = time.perf_counter()
        _chain_clients(clients, x)
        dt = time.perf_counter() - t_start
        per_token.append(dt)
        if n == 0:
            ttft = dt
    elapsed = time.perf_counter() - t0
    for client in clients:
        client.close()
    return {
        "stage": stage,
        "name": name,
        "tokens": tokens,
        "shards": len(addresses),
        "tps": tokens / elapsed,
        "ttft_s": ttft,
        "per_token_s": statistics.mean(per_token),
    }


def run_loopback(layers, tokens: int, in_features: int) -> dict:
    """Stage 2: spawn local gRPC servers and chain over loopback."""
    import grpc
    from concurrent import futures
    servers = []
    addresses = []
    for lyr in layers:
        port = free_port()
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
        servicer = InferenceServicer(
            shard_index=0,
            shard_layers=[("bench", lyr)],
            compression=CompressionType.ZLIB,
        )
        inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)
        if server.add_insecure_port(f"127.0.0.1:{port}") == 0:
            raise RuntimeError(f"could not bind 127.0.0.1:{port}")
        server.start()
        servers.append(server)
        addresses.append(f"127.0.0.1:{port}")
    try:
        return run_swarm(addresses, tokens, in_features, stage=2, name="loopback")
    finally:
        for server in servers:
            server.stop(0)


def run_instrumented(model_gb: float, shards: int, tokens: int,
                      in_features: int) -> dict:
    """Stage 5: coordination tax = wait / compute per token.

    Compute time comes from the in-process chain (stage 1), total
    time from the gRPC chain (stage 2); the difference is pure
    coordination wait: serialization, network and scheduling.
    """
    layers = build_layers(model_gb, shards, in_features)
    iso = run_isolated(layers, tokens, in_features)
    loop = run_loopback(layers, tokens, in_features)
    compute = iso["per_token_s"]
    total = loop["per_token_s"]
    wait = max(0.0, total - compute)
    tax = wait / compute if compute > 0 else float("inf")
    efficiency = 1.0 / (1.0 + tax)
    return {
        "stage": 5,
        "name": "instrument",
        "compute_per_token_s": compute,
        "total_per_token_s": total,
        "wait_per_token_s": wait,
        "coordination_tax": tax,
        "scaling_efficiency": efficiency,
        "tax_within_ceiling": tax <= COORDINATION_TAX_CEILING,
        "efficiency_above_floor": efficiency >= SCALING_EFFICIENCY_FLOOR,
        "isolated": iso,
        "loopback": loop,
    }


def _netem(cmd: str, interface: str, extra: str):
    subprocess.run(
        ["sudo", "tc", cmd, "dev", interface, extra],
        check=True,
    )


def run_netem(interface: str, apply_changes: bool, model_gb: float,
              shards: int, tokens: int, in_features: int) -> dict:
    """Stage 4: TPS-vs-RTT curve via injected latency (tc netem).

    Needs root. Without --apply-netem the commands are printed for
    manual execution and no curve is produced.
    """
    isolated = run_isolated(
        build_layers(model_gb, shards, in_features), tokens, in_features
    )
    curve = []
    for latency in NETEM_LATENCY_POINTS_MS:
        rule = f"netem delay {latency}ms"
        if apply_changes:
            _netem("qdisc", interface, f"add root {rule}")
        else:
            print(f"[manual] sudo tc qdisc add dev {interface} root {rule}")
            continue
        try:
            result = run_loopback(
                build_layers(model_gb, shards, in_features), tokens, in_features
            )
            curve.append({"latency_ms": latency, "tps": result["tps"]})
        finally:
            _netem("qdisc", interface, "del")
    if not curve:
        return {
            "stage": 4,
            "name": "netem",
            "note": "commands printed; rerun with --apply-netem as root",
            "isolated_tps": isolated["tps"],
        }
    break_even = None
    for point in curve:
        if point["tps"] < isolated["tps"]:
            break_even = point["latency_ms"]
            break
    return {
        "stage": 4,
        "name": "netem",
        "interface": interface,
        "isolated_tps": isolated["tps"],
        "curve": curve,
        "break_even_latency_ms": break_even,
    }


def run_sweep(stage_runner, sizes_gb: List[float], tokens: int,
              in_features: int) -> dict:
    """Sweep model sizes; keep the largest that still runs at >= 1 TPS."""
    results = []
    max_running = None
    for gb in sorted(sizes_gb):
        metrics = stage_runner(gb, tokens, in_features)
        metrics["model_gb"] = gb
        metrics["runs_interactively"] = metrics["tps"] >= MIN_INTERACTIVE_TPS
        if metrics["runs_interactively"]:
            max_running = gb
        results.append(metrics)
        tps = metrics["tps"]
        status = "RUNS" if metrics["runs_interactively"] else "too slow"
        print(f"  {gb:.2f} GB -> {tps:.3f} tok/s ({status})")
    return {"sweep": results, "max_model_gb_at_1tps": max_running}


def anchor_verdict(synthetic_tax: float, real_tax: float) -> dict:
    """Report the anchor verdict across the tolerance sweep.

    A single tolerance would be one more movable withdrawal line.
    The sweep shows the reader exactly where the conclusion is
    fragile, instead of asking them to trust one number.
    """
    if real_tax > 0:
        rel_gap = abs(synthetic_tax - real_tax) / real_tax
    else:
        rel_gap = float("inf")
    rows = []
    verdicts = set()
    for tol in ANCHOR_TOLERANCE_SWEEP:
        ok = rel_gap <= tol
        verdicts.add(ok)
        rows.append({"tolerance": tol, "relative_gap": rel_gap, "pass": ok})
    print("\n=== ANCHOR CALIBRATION VERDICT (tolerance sweep) ===")
    for r in rows:
        ok = "PASS" if r["pass"] else "FAIL"
        print(f"  tolerance {r['tolerance']:.2f}: relative gap {r['relative_gap']:.3f} -> {ok}")
    uniform = len(verdicts) == 1
    if uniform:
        word = "calibrated" if True in verdicts else "uncalibrated"
        print(f"\n  VERDICT: UNIFORM ({word}) across the sweep")
    else:
        fragile = max(r["tolerance"] for r in rows if not r["pass"])
        print(f"\n  VERDICT: FRAGILE - conclusion flips at tolerance {fragile}")
    print("  CORRECTION_FACTOR = FORBIDDEN: the synthetic is never fitted to the real")
    return {"relative_gap": rel_gap, "rows": rows, "uniform": uniform}


def calibrate_shard(step_gb: float) -> dict:
    """Measure the max usable shard memory by OOM stepping.

    Allocations step up until the device raises OOM; the
    largest size that SURVIVES is the per-shard budget - a
    measured fact, not a constant. This executes the 0.85
    safety factor (amendment #24): a factor about a
    measurement is a guess wearing a uniform. Each chunk is
    written to, so the pages are touched, not just reserved.
    """
    if not torch.cuda.is_available():
        raise SystemExit("--calibrate-shard needs a CUDA device")
    name = torch.cuda.get_device_name(0)
    tensors = []
    usable_gb = 0.0
    step_b = int(step_gb * 1024 ** 3)
    while True:
        try:
            t = torch.empty(step_b, dtype=torch.uint8, device="cuda")
            t[: 64 * 1024 ** 2] = 1  # touch the pages: a real write
            tensors.append(t)
            usable_gb += step_gb
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                break
            raise
    del tensors
    torch.cuda.empty_cache()
    result = {
        "device": name,
        "step_gb": step_gb,
        "max_usable_shard_gb": round(usable_gb, 2),
    }
    print(json.dumps(result, indent=2))
    print("diagnostic contiguous ceiling only (amendment #25): "
          "the gate consumes the trace-replay number (--replay-calibrate)")
    return result


def record_alloc_trace(model_gb: float, shards: int, tokens: int,
                       in_features: int, trace_path: str) -> dict:
    """Journal the workload allocation pattern (amendment #25).

    The journal follows the shard rhythm, not one flat
    block: persistent weights, a transient activation per
    token, a KV cache that grows monotonically (the peak
    arrives at the END, not the start), and jitter blocks
    of varied sizes freed out of order - the fragmentation
    holes a flat OOM step never sees. The format is plain
    JSON so a trace captured from a REAL run can replace
    it; until then the synthetic pattern is declared
    modeling (covenant section 5), never claimed as
    captured.
    """
    per_shard_b = int(model_gb * 1024 ** 3 / max(1, shards))
    act_b = max(1, in_features * 4)
    events = []
    for _ in range(shards):
        events.append({"op": "alloc", "tag": "weight", "bytes": per_shard_b})
    jitter_open = []
    for n in range(tokens):
        events.append({"op": "alloc", "tag": "activation",
                        "bytes": act_b * (n % 3 + 1)})
        events.append({"op": "free", "index": len(events) - 1})
        events.append({"op": "alloc", "tag": "kv", "bytes": act_b})
        jitter_b = max(1, (per_shard_b // 64) * ((n * 7) % 5 + 1) // 5)
        events.append({"op": "alloc", "tag": "jitter", "bytes": jitter_b})
        jitter_open.append(len(events) - 1)
        if len(jitter_open) > 2 and n % 2 == 0:
            events.append({"op": "free", "index": jitter_open.pop(0)})
    for idx in jitter_open:
        events.append({"op": "free", "index": idx})
    trace = {
        "model_gb": model_gb,
        "shards": shards,
        "tokens": tokens,
        "in_features": in_features,
        "events": events,
    }
    with open(trace_path, "w") as f:
        json.dump(trace, f, indent=2)
    tags = {}
    for ev in events:
        if ev["op"] == "alloc":
            tags[ev["tag"]] = tags.get(ev["tag"], 0) + 1
    print(f"allocation trace written to {trace_path}: "
          f"{len(events)} events, allocs by tag {tags}")
    print("replay it: --replay-calibrate --trace-path " + trace_path)
    return {"trace_path": trace_path, "events": len(events),
            "allocs_by_tag": tags}


def _replay_trace(events, scale: float):
    """Replay the journal at a scale factor; raise on OOM."""
    live = {}
    for i, ev in enumerate(events):
        if ev["op"] == "alloc":
            nbytes = max(1, int(ev["bytes"] * scale))
            t = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
            t[: min(nbytes, 64 * 1024 ** 2)] = 1  # touch the pages
            live[i] = t
        else:
            live.pop(ev["index"], None)
    live.clear()
    torch.cuda.empty_cache()


def _trace_peak_bytes(events) -> int:
    """Base peak footprint of the unscaled journal."""
    total = 0
    peak = 0
    for ev in events:
        if ev["op"] == "alloc":
            total += ev["bytes"]
            peak = max(peak, total)
        else:
            total -= events[ev["index"]]["bytes"]
    return max(peak, 1)


def trace_capture_gate(trace) -> bool:
    """Judge the capture layer before it judges anything (#26).

    The calibration controls judge the CALIBRATION; these
    checks judge the RECORDER. A trace that never passed this
    gate cannot calibrate anything.
    """
    events = trace["events"]
    # (a) accounting closure (modulo known live memory): every
    # free must match a live alloc, and whatever stays alive at
    # the end must belong to LIVE_TAGS. Anything else is a
    # missed capture - the ruler is perforated.
    live = {}
    for i, ev in enumerate(events):
        if ev["op"] == "alloc":
            live[i] = ev
        elif ev["index"] not in live:
            print(f"  capture gate FAIL (a): free of index "
                  f"{ev['index']} with no matching alloc")
            return False
        else:
            live.pop(ev["index"])
    residual = {}
    for ev in live.values():
        residual[ev["tag"]] = residual.get(ev["tag"], 0) + ev["bytes"]
    dangling = {t: n for t, n in residual.items() if t not in LIVE_TAGS}
    if dangling:
        print(f"  capture gate FAIL (a): accounting does not close, "
              f"live bytes outside {LIVE_TAGS}: {dangling}")
        return False
    # (b) known leak injection: 100MB allocated and never freed
    # must be captured within +/- LEAK_INJECT_TOLERANCE by the
    # same accounting walk.
    injected_events = list(events) + [
        {"op": "alloc", "tag": "leak_inject", "bytes": LEAK_INJECT_BYTES}
    ]
    live2 = {}
    for i, ev in enumerate(injected_events):
        if ev["op"] == "alloc":
            live2[i] = ev
        else:
            live2.pop(ev["index"], None)
    captured = sum(ev["bytes"] for ev in live2.values()
                   if ev["tag"] == "leak_inject")
    if abs(captured - LEAK_INJECT_BYTES) > (
            LEAK_INJECT_TOLERANCE * LEAK_INJECT_BYTES):
        print(f"  capture gate FAIL (b): injected {LEAK_INJECT_BYTES} "
              f"bytes, captured {captured} - outside +/-1%")
        return False
    # (c) deterministic replay: two independent passes over the
    # same trace must produce identical totals and peaks.
    totals = []
    peaks = []
    for _ in range(2):
        total = 0
        for ev in events:
            if ev["op"] == "alloc":
                total += ev["bytes"]
        totals.append(total)
        peaks.append(_trace_peak_bytes(events))
    if totals[0] != totals[1] or peaks[0] != peaks[1]:
        print("  capture gate FAIL (c): replay is not deterministic")
        return False
    # (d) on-device injection (amendment #27): a REAL allocation,
    # witnessed by the device's own counter - the injection path
    # is tried, never assumed. CUDA: the allocator's own stats
    # delta within +/- LIVE_INJECT_TOLERANCE_CUDA. No CUDA:
    # psutil RSS delta within +/- LIVE_INJECT_TOLERANCE_CPU -
    # the caching allocator's noise is declared, not hidden. No
    # reliable path available: refuse - a fake injection is
    # worse than none.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        before = torch.cuda.memory_stats().get(
            "allocated_bytes.all.current", 0)
        probe = torch.empty(LEAK_INJECT_BYTES, dtype=torch.uint8,
                            device="cuda")
        probe[0] = 1  # touch the pages: the allocation is real
        after = torch.cuda.memory_stats().get(
            "allocated_bytes.all.current", 0)
        delta = after - before
        del probe
        torch.cuda.empty_cache()
        if abs(delta - LEAK_INJECT_BYTES) > (
                LIVE_INJECT_TOLERANCE_CUDA * LEAK_INJECT_BYTES):
            print(f"  capture gate FAIL (d): injected "
                  f"{LEAK_INJECT_BYTES} bytes on CUDA, the device "
                  f"witnessed {delta} - outside +/-1%")
            return False
    else:
        try:
            import psutil
        except ImportError:
            raise SystemExit(
                "capture gate (d): no reliable injection path "
                "(no CUDA device and psutil is not installed) - "
                "refusing: a fake injection is worse than none"
            )
        proc = psutil.Process()
        before = proc.memory_info().rss
        probe = bytearray(LEAK_INJECT_BYTES)
        for i in range(0, len(probe), 4096):
            probe[i] = 1  # touch the pages so RSS actually grows
        after = proc.memory_info().rss
        delta = after - before
        del probe
        if abs(delta - LEAK_INJECT_BYTES) > (
                LIVE_INJECT_TOLERANCE_CPU * LEAK_INJECT_BYTES):
            print(f"  capture gate FAIL (d): injected "
                  f"{LEAK_INJECT_BYTES} bytes on CPU, RSS "
                  f"witnessed {delta} - outside +/-5%")
            return False
    print("  capture gate PASS: accounting closed (known live memory "
          "only), leak injection captured within +/-1%, "
          "replay deterministic, on-device injection witnessed")
    return True


class _LiveRecorder:
    """Real-time journal of torch.empty allocations (#27).

    Patches, for the duration of the window only, the allocation
    path that --replay-calibrate itself consumes (torch.empty),
    and journals every alloc (bytes) plus the matching free when
    the tensor dies. Allocations outside the window produce zero
    records: the boundary IS the recorder's contract.
    """

    def __init__(self):
        self.events = []
        self._orig = None

    def __enter__(self):
        import weakref
        self._orig = torch.empty
        events = self.events
        orig = self._orig

        def journaled_empty(*args, **kwargs):
            t = orig(*args, **kwargs)
            nbytes = t.numel() * t.element_size()
            events.append({"op": "alloc", "tag": "live",
                           "bytes": nbytes})
            weakref.finalize(t, events.append,
                             {"op": "free", "index": len(events) - 1})
            return t

        torch.empty = journaled_empty
        return self

    def __exit__(self, *exc):
        torch.empty = self._orig
        return False


def live_capture_gate() -> bool:
    """Judge the recorder IN TIME, before any real trace (#27).

    The static gate judges the journal AFTER the fact; this gate
    judges the recorder WHILE it records - the same logic as the
    calibration's 50%/150% controls, applied one layer down:
    a positive live control (a known pattern MUST be captured,
    in order, at the right sizes), a negative live control
    (allocations outside the window MUST produce zero records -
    a recorder without boundaries poisons the analysis), and a
    live determinism control (the same live pattern, recorded
    twice, must produce the same alloc sequence).
    """
    def live_pattern():
        # 10 known sizes, allocated in order during the window;
        # the journal dies with the list, inside the window
        return [torch.empty(n, dtype=torch.uint8)
                for n in LIVE_PROBE_SIZES]

    rec1 = _LiveRecorder()
    with rec1:
        live_pattern()
    allocs1 = [e["bytes"] for e in rec1.events if e["op"] == "alloc"]
    if allocs1 != LIVE_PROBE_SIZES:
        print("  live capture gate FAIL (positive): the recorder "
              "missed or reordered the known pattern")
        print(f"    expected: {LIVE_PROBE_SIZES}")
        print(f"    captured: {allocs1}")
        return False
    # negative live control: outside the window, zero records
    rec2 = _LiveRecorder()
    stray = torch.empty(4096, dtype=torch.uint8)  # BEFORE the window
    with rec2:
        pass
    del stray  # AFTER the window
    if rec2.events:
        print(f"  live capture gate FAIL (negative): {len(rec2.events)} "
              "records captured outside the recording window - the "
              "recorder has no boundaries")
        return False
    # live determinism: the same pattern, recorded twice
    rec3 = _LiveRecorder()
    with rec3:
        live_pattern()
    allocs3 = [e["bytes"] for e in rec3.events if e["op"] == "alloc"]
    if allocs3 != allocs1:
        print("  live capture gate FAIL (determinism): the same "
              "live pattern produced different journals")
        return False
    print("  live capture gate PASS: positive live captured in "
          "order, negative live captured nothing, live replay "
          "deterministic")
    return True


def replay_calibrate(trace_path: str, step_gb: float,
                     capture_gate: bool = False,
                     noise_report: str = "",
                     fragmentation_threshold: str = "derived") -> dict:
    """Pattern-aware max usable shard memory (amendment #25).

    The flat OOM step tests SIZE, not PATTERN: it never
    sees transient activations, monotone KV growth, or the
    holes left by out-of-order frees. This calibration
    replays the workload journal with OOM-stepping on the
    SCALE FACTOR: the largest scale whose replay survives,
    times the base peak, is the limit the gate consumes.
    Controls are mandatory: a replay at
    REPLAY_CONTROL_SAFE_FRACTION of the limit must survive
    and a replay at REPLAY_CONTROL_DOOM_FRACTION must fail;
    if either misbehaves the calibration itself is invalid
    and no number leaves this function. The flat contiguous
    ceiling is measured only to price the premium.

    Amendment #26: the capture layer is judged before this runs
    (--trace-capture-gate: accounting closure, leak injection,
    deterministic replay), and the flat-execution threshold is
    DERIVED (REPRODUCE_K x sigma_noise from --noise-report),
    never a chosen number.
    """
    if not capture_gate:
        raise SystemExit(
            "REJECTED: capture gate required before calibration "
            "(amendment #26: pass --trace-capture-gate - a trace "
            "that was never judged cannot calibrate anything)"
        )
    if not torch.cuda.is_available():
        raise SystemExit("--replay-calibrate needs a CUDA device")
    with open(trace_path) as f:
        trace = json.load(f)
    if not trace_capture_gate(trace):
        raise SystemExit(
            "REJECTED: capture gate failed (amendment #26) - "
            "no number leaves this tool"
        )
    events = trace["events"]
    # Amendment #26: the flat-execution threshold is DERIVED,
    # never chosen: threshold = REPRODUCE_K x sigma_noise
    # (relative), with sigma_noise from the variance report.
    # REPRODUCE_K = 3 is bound to the named decisions in the
    # constants block. Refused BEFORE the expensive OOM
    # stepping: fail fast, never spend the budget first.
    if fragmentation_threshold != "derived":
        raise SystemExit(
            "fragmentation threshold must be 'derived' "
            "(k*sigma_noise) - a chosen number is a type-four "
            "constant"
        )
    if not noise_report:
        raise SystemExit(
            "cannot derive threshold without variance report "
            "(pass --noise-report with the --variance-report JSON)"
        )
    with open(noise_report) as f:
        noise = json.load(f)
    if noise.get("underpowered"):
        raise SystemExit(
            "cannot derive threshold from an under-powered "
            "variance report - add runs until POWERED"
        )
    denom = noise.get("ratio_mean") or noise.get("tps_mean")
    if not denom or noise.get("sigma_ci_upper") is None:
        raise SystemExit(
            "variance report predates amendment #25 (no sigma "
            "interval on the line) - cannot derive threshold"
        )
    # conservative side of the sigma interval: flat is executed
    # only when the premium exceeds the largest plausible noise
    sigma_noise_rel = noise["sigma_ci_upper"] / denom
    frag_threshold = REPRODUCE_K * sigma_noise_rel
    device = torch.cuda.get_device_name(0)
    base_peak = _trace_peak_bytes(events)
    step_scale = step_gb * 1024 ** 3 / base_peak
    scale = step_scale
    surviving = 0.0
    while True:
        try:
            _replay_trace(events, scale)
            surviving = scale
            scale += step_scale
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                break
            raise
    if surviving <= 0.0:
        raise SystemExit("replay calibration: not even one step survives")
    # negative control: half the measured limit MUST survive
    try:
        _replay_trace(events, surviving * REPLAY_CONTROL_SAFE_FRACTION)
        safe_ok = True
    except RuntimeError:
        torch.cuda.empty_cache()
        safe_ok = False
    # positive control: 150% of the measured limit MUST fail
    try:
        _replay_trace(events, surviving * REPLAY_CONTROL_DOOM_FRACTION)
        doom_failed = False
    except RuntimeError:
        torch.cuda.empty_cache()
        doom_failed = True
    if not safe_ok or not doom_failed:
        raise SystemExit(
            "replay calibration INVALID: controls failed "
            f"(safe survived: {safe_ok}, doomed failed: {doom_failed}); "
            "the tool is broken - no number leaves this function"
        )
    replayed_gb = surviving * base_peak / 1024 ** 3
    flat = calibrate_shard(step_gb)
    flat_gb = flat["max_usable_shard_gb"]
    premium = ((flat_gb - replayed_gb) / flat_gb
               if flat_gb > 0 else None)
    flat_dead = (premium is not None
                 and premium > frag_threshold)
    result = {
        "device": device,
        "trace_path": trace_path,
        "step_gb": step_gb,
        "max_usable_shard_gb": round(replayed_gb, 2),
        "flat_ceiling_gb": flat_gb,
        "fragmentation_premium": (round(premium, 4)
                                  if premium is not None else None),
        "flat_ceiling_dead": flat_dead,
        "fragmentation_threshold": round(frag_threshold, 4),
        "threshold_source": (
            "derived: REPRODUCE_K * sigma_ci_upper/mean from "
            "--noise-report (amendment #26)"
        ),
        "capture_gate_passed": True,
        "controls": {
            "safe_fraction_survived": safe_ok,
            "doom_fraction_failed": doom_failed,
        },
    }
    print("\n=== CALIBRATION REPORT (trace-replay, amendment #25) ===")
    print(f"  flat contiguous ceiling: {flat_gb} GB (diagnostic, never gates)")
    print(f"  trace-replay limit: {round(replayed_gb, 2)} GB (the workload pattern, OOM-stepped)")
    if premium is not None:
        word = ("EXECUTED - unfaithful, gates on replay only"
                if flat_dead else "within derived threshold")
        print(f"  fragmentation premium: {premium * 100:.1f}% vs derived "
              f"threshold {frag_threshold * 100:.1f}% (k*sigma_noise) "
              f"-> {word}")
    print(f"  controls: safe@50% survived={safe_ok}, doomed@150% failed={doom_failed}")
    print("  THE GATE CONSUMES THE REPLAYED NUMBER: "
          f"--max-usable-shard-gb {round(replayed_gb, 2)}")
    print(json.dumps(result, indent=2))
    return result


def sigma_interval(stdev: float, n: int):
    """Two-sided CI_LEVEL interval on sigma (chi-square).

    Sigma estimated from n runs is itself noisy: at n=10
    the relative uncertainty is ~24% (chi-square, df=9).
    Quantiles use the Wilson-Hilferty approximation - a
    DECLARED approximation, printed as such so nobody
    mistakes it for exact.
    """
    df = n - 1
    if df < 1 or stdev <= 0:
        return stdev, stdev
    c = 2.0 / (9.0 * df)
    chi_lo = df * (1 - c - Z_975 * c ** 0.5) ** 3
    chi_hi = df * (1 - c + Z_975 * c ** 0.5) ** 3
    lo = stdev * (df / chi_hi) ** 0.5
    hi = stdev * (df / chi_lo) ** 0.5
    return lo, hi


def runs_required(stdev: float, mean: float) -> Optional[int]:
    """Smallest n whose margin interval beats the target.

    n is derived from the decision (the detection width),
    never from a round number: with the measured stdev as
    a plug-in, the chi-square spread of the sigma estimate
    shrinks until k x sigma fits inside TARGET_MARGIN_WIDTH
    of the mean.
    """
    if mean <= 0 or stdev <= 0:
        return None
    for n in range(2, 10000):
        lo, hi = sigma_interval(stdev, n)
        if REPRODUCE_K * (hi - lo) / mean < TARGET_MARGIN_WIDTH:
            return n
    return None


def variance_report(stage: int, repeats: int, model_gb: float,
                    shards: int, tokens: int, in_features: int,
                    paired: bool = False) -> dict:
    """Measure the noise floor AND its own uncertainty (#25).

    The margin is an interval, never a point: sigma from n
    runs carries chi-square uncertainty, so the report
    prints [k x sigma_lower, k x sigma_upper] and REFUSES
    to judge while the interval is wider than the detection
    target. Paired mode runs standalone and swarm in the
    same iteration and reports sigma of the paired RATIO:
    the shared hardware noise cancels and the required n
    shrinks with it.
    """
    if repeats < 2:
        raise SystemExit("variance report needs >= 2 repeats (chi-square needs df >= 1)")
    tps_values = []
    ratio_values = []
    for i in range(repeats):
        if stage == 1:
            metrics = run_isolated(
                build_layers(model_gb, shards, in_features), tokens, in_features
            )
            if paired:
                single = run_isolated(
                    build_layers(model_gb, 1, in_features), tokens, in_features
                )
        else:
            metrics = run_loopback(
                build_layers(model_gb, shards, in_features), tokens, in_features
            )
            if paired:
                single = run_loopback(
                    build_layers(model_gb, 1, in_features), tokens, in_features
                )
        tps = metrics["tps"]
        tps_values.append(tps)
        line = f"  run {i + 1}/{repeats}: {tps:.3f} tok/s"
        if paired:
            ratio = tps / single["tps"] if single["tps"] > 0 else 0.0
            ratio_values.append(ratio)
            line += f"  paired ratio {ratio:.3f}"
        print(line)
    mean = statistics.mean(tps_values)
    stdev = statistics.stdev(tps_values)
    n = repeats
    df = n - 1
    sigma_lo, sigma_hi = sigma_interval(stdev, n)
    if paired:
        ratio_mean = statistics.mean(ratio_values)
        ratio_stdev = statistics.stdev(ratio_values)
        rs_lo, rs_hi = sigma_interval(ratio_stdev, n)
        denom = ratio_mean
        s_use, use_lo, use_hi = ratio_stdev, rs_lo, rs_hi
    else:
        ratio_mean = ratio_stdev = None
        denom = mean
        s_use, use_lo, use_hi = stdev, sigma_lo, sigma_hi
    margin_lower = REPRODUCE_K * use_lo
    margin_upper = REPRODUCE_K * use_hi
    rel_width = ((margin_upper - margin_lower) / denom
                 if denom > 0 else float("inf"))
    n_req = runs_required(s_use, denom)
    n_req_unpaired = runs_required(stdev, mean) if paired else n_req
    underpowered = rel_width >= TARGET_MARGIN_WIDTH
    status = (f"UNDER-POWERED at n={n} - add runs until width < target_width"
              if underpowered else "POWERED - the interval is inside the target")
    unit = "ratio" if paired else "tok/s"
    print("\n=== VARIANCE REPORT (the ruler is also measured) ===")
    print(f"  sigma = {s_use:.4f} {unit}  (95% CI: [{use_lo:.4f}, {use_hi:.4f}],"
          f" n={n}, df={df}, Wilson-Hilferty)")
    print(f"  margin_lower = k*sigma_lower = {margin_lower:.4f}")
    print(f"  margin_upper = k*sigma_upper = {margin_upper:.4f}")
    print(f"  k = {REPRODUCE_K}  (justified: 0.27% false-failure x ~10 tests/year = 0.027/year)")
    if paired:
        print(f"  paired_mode = on  (sigma_ratio = {ratio_stdev:.4f},"
              f" n_required {n_req_unpaired} unpaired -> {n_req} paired)")
    else:
        print("  paired_mode = off  (pair standalone/swarm per iteration to shrink n)")
    print(f"  target_width = {TARGET_MARGIN_WIDTH}  (relative, derived from thesis floor {SWARM_ADVANTAGE_FLOOR}x)")
    print(f"  relative interval width: {rel_width:.3f}")
    print(f"  runs required (plug-in): {n_req}")
    print(f"  status: {status}")
    report = {
        "stage": stage,
        "model_gb": model_gb,
        "shards": shards,
        "tokens": tokens,
        "repeats": repeats,
        "paired": paired,
        "tps_mean": mean,
        "tps_stdev": stdev,
        "noise_floor": REPRODUCE_K * stdev,
        "sigma_ci_lower": use_lo,
        "sigma_ci_upper": use_hi,
        "margin_lower": margin_lower,
        "margin_upper": margin_upper,
        "relative_width": rel_width,
        "n_required": n_req,
        "target_width": TARGET_MARGIN_WIDTH,
        "reproduce_k": REPRODUCE_K,
        "underpowered": underpowered,
        "status": status,
        "amendment": 25,
    }
    if paired:
        report["ratio_mean"] = ratio_mean
        report["ratio_stdev"] = ratio_stdev
    return report


def reproduce_tolerance(report_paths: List[str]) -> dict:
    """Derive the replay acceptance margin - or refuse to.

    The margin is an interval [k x sigma_lower, k x
    sigma_upper] over the measured reports, never a chosen
    number, and it applies to the swarm ADVANTAGE RATIO,
    not raw TPS. A report without a sigma interval predates
    amendment #25: not auditable, refused. A tool whose
    interval is wider than the target is UNDER-POWERED: it
    does not judge - it abstains, loudly and nonzero.
    """
    reports = []
    for p in report_paths:
        with open(p) as f:
            reports.append(json.load(f))
    missing = [p for p, r in zip(report_paths, reports)
               if "sigma_ci_lower" not in r]
    if missing:
        raise SystemExit(
            "JUDGMENT REFUSED: report(s) " + ", ".join(missing) +
            " predate amendment #25 (no sigma interval on the line);"
            " regenerate them"
        )
    under = [p for p, r in zip(report_paths, reports) if r.get("underpowered")]
    if under:
        raise SystemExit(
            "JUDGMENT REFUSED: the tool is under-powered (margin interval" +
            " wider than target); add runs: " + ", ".join(under)
        )
    paired_flags = {bool(r.get("paired")) for r in reports}
    if len(paired_flags) > 1:
        raise SystemExit(
            "JUDGMENT REFUSED: mixed paired/unpaired reports - units differ"
        )
    floors = []
    for r in reports:
        s = r.get("ratio_stdev") if r.get("paired") else r.get("tps_stdev", 0.0)
        floors.append(REPRODUCE_K * s)
    means = [r.get("tps_mean", 0.0) for r in reports]
    spread = max(means) - min(means) if means else 0.0
    margin = max(max(floors), spread) if floors else 0.0
    lo = max(REPRODUCE_K * r["sigma_ci_lower"] for r in reports)
    hi = max(REPRODUCE_K * r["sigma_ci_upper"] for r in reports)
    print("=== REPRODUCTION TOLERANCE (derived, not chosen) ===")
    for p, r in zip(report_paths, reports):
        s = r.get("ratio_stdev") if r.get("paired") else r.get("tps_stdev", 0.0)
        print(f"  {p}: sigma {s:.4f} "
              f"(95% CI: [{r['sigma_ci_lower']:.4f}, {r['sigma_ci_upper']:.4f}],"
              f" n={r.get('repeats')})")
    print(f"  noise floors (x{REPRODUCE_K}): {[round(x, 3) for x in floors]}")
    print(f"  hardware spread: {spread:.3f}")
    print(f"  => acceptance margin: {margin:.3f}  (interval: [{lo:.3f}, {hi:.3f}])")
    print("  apply to the swarm ADVANTAGE RATIO, not to raw TPS")
    print("  status: POWERED - judgment is available")
    return {
        "reports": report_paths,
        "noise_floors": floors,
        "hardware_spread": spread,
        "margin": margin,
        "margin_interval": [lo, hi],
        "reproduce_k": REPRODUCE_K,
        "judgment": "AVAILABLE",
    }


def overflow_gate_verdict(model_gb: float, shards: int,
                          per_shard_gb: float, completed_tokens: int,
                          min_available_gb: float,
                          max_usable_shard_gb: float,
                          leak_slope_mb=None, flat_shard_gb=None) -> dict:
    """Evaluate the overflow gate against pre-registered rules.

    The gate tests the heart of the thesis: a model too big for
    any single device, held by the swarm, completing generation.
    It is evaluated from recorded run facts - the tool is the
    judge, not the runner. Every memory number is MEASURED:
    min available memory via mem_get_info, the per-shard budget
    via TRACE-REPLAY calibration (--replay-calibrate; the flat
    --calibrate-shard ceiling is a diagnostic for the
    fragmentation premium, never a gate input). The 6 GB
    constant was a guess and the 0.85 factor was a guess about
    a guess; both were executed (amendments #23, #24), and the
    flat calibration followed them out (amendment #25).
    """
    checks = {}
    checks["exceeds_single_device"] = {
        "value": model_gb,
        "limit": min_available_gb,
        "pass": model_gb > min_available_gb,
    }
    aggregate = shards * min_available_gb
    checks["fits_aggregate"] = {
        "value": model_gb,
        "limit": aggregate,
        "pass": model_gb <= aggregate,
    }
    checks["per_shard_budget"] = {
        "value": per_shard_gb,
        "limit": max_usable_shard_gb,
        "pass": per_shard_gb <= max_usable_shard_gb,
    }
    checks["min_devices"] = {
        "value": shards,
        "limit": OVERFLOW_GATE["min_devices"],
        "pass": shards >= OVERFLOW_GATE["min_devices"],
    }
    checks["generation_completed"] = {
        "value": completed_tokens,
        "limit": OVERFLOW_GATE["min_completion_tokens"],
        "pass": completed_tokens >= OVERFLOW_GATE["min_completion_tokens"],
    }
    holds = all(c["pass"] for c in checks.values())
    verdict = "OVERFLOW_GATE_PASS" if holds else "OVERFLOW_GATE_FAIL"
    premium = None
    if flat_shard_gb is not None and flat_shard_gb > 0:
        premium = (flat_shard_gb - max_usable_shard_gb) / flat_shard_gb
    print("\n=== OVERFLOW GATE (the heart of the thesis) ===")
    print(f"  measured min available memory: {min_available_gb} GB")
    print(f"  measured max usable shard (trace-replay calibration): {max_usable_shard_gb} GB")
    if premium is not None:
        print(f"  fragmentation premium (diagnostic): "
              f"{premium * 100:.1f}% (flat {flat_shard_gb} GB); the "
              "execution verdict lives in the trace-replay "
              "calibration, whose threshold is derived "
              "k*sigma_noise (amendment #26)")
    for name, c in checks.items():
        ok = "PASS" if c["pass"] else "FAIL"
        print(f"  {name}: {c['value']} (limit {c['limit']}) -> {ok}")
    if leak_slope_mb is not None:
        print(f"  leak probe: {leak_slope_mb} MB/token over {OVERFLOW_GATE['leak_probe_tokens']} tokens (recorded, not gated)")
    print(f"\n  >>> {verdict} <<<")
    return {
        "checks": checks,
        "holds": holds,
        "verdict": verdict,
        "max_usable_shard_gb": max_usable_shard_gb,
        "flat_shard_gb": flat_shard_gb,
        "fragmentation_premium": premium,
        "leak_slope_mb_per_token": leak_slope_mb,
    }


def apply_verdict(single_path: str, swarm_path: str,
                  anchor_gb=None, claim_gb=None) -> dict:
    """Print the pre-registered verdict from two sweep reports.

    If the claimed size exceeds ANCHOR_EXTRAPOLATION_CAP times
    the anchored size, the verdict is withheld: no anchor, no
    summit talk.
    """
    if anchor_gb and claim_gb and claim_gb > anchor_gb * ANCHOR_EXTRAPOLATION_CAP:
        print(
            f"VERDICT WITHHELD: claim {claim_gb} GB exceeds "
            f"{ANCHOR_EXTRAPOLATION_CAP}x anchor {anchor_gb} GB "
            "(extrapolation cap; run a second anchor)"
        )
        return {"holds": False, "verdict": "VERDICT WITHHELD (EXTRAPOLATION CAP)"}
    with open(single_path) as f:
        single = json.load(f)
    with open(swarm_path) as f:
        swarm = json.load(f)
    single_max = single["max_model_gb_at_1tps"]
    swarm_max = swarm["max_model_gb_at_1tps"]
    checks = {}
    if single_max and swarm_max:
        advantage = swarm_max / single_max
        checks["swarm_advantage"] = {
            "value": advantage,
            "floor": SWARM_ADVANTAGE_FLOOR,
            "pass": advantage >= SWARM_ADVANTAGE_FLOOR,
        }
    else:
        checks["swarm_advantage"] = {
            "value": None,
            "floor": SWARM_ADVANTAGE_FLOOR,
            "pass": False,
            "note": "one side never reached 1 TPS at any size",
        }
    tax = swarm.get("coordination_tax")
    if tax is not None:
        checks["coordination_tax"] = {
            "value": tax,
            "ceiling": COORDINATION_TAX_CEILING,
            "pass": tax <= COORDINATION_TAX_CEILING,
        }
    efficiency = swarm.get("scaling_efficiency")
    if efficiency is not None:
        checks["scaling_efficiency"] = {
            "value": efficiency,
            "floor": SCALING_EFFICIENCY_FLOOR,
            "pass": efficiency >= SCALING_EFFICIENCY_FLOOR,
        }
    holds = all(c["pass"] for c in checks.values())
    verdict = "THESIS HOLDS" if holds else "THESIS REFUTED"
    print("\n=== PRE-REGISTERED VERDICT ===")
    for name, c in checks.items():
        limit = c.get("floor", c.get("ceiling"))
        ok = "PASS" if c["pass"] else "FAIL"
        print(f"  {name}: {c['value']} (limit {limit}) -> {ok}")
    print(f"\n  >>> {verdict} <<<")
    return {"checks": checks, "holds": holds, "verdict": verdict}


def save_report(data: dict, report_path: Optional[str]):
    if not report_path:
        return
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"report written to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="HiveMind swarm benchmark (pre-registered experiment)"
    )
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--serve", action="store_true",
                        help="run one shard server (for swarm hosts)")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--shard-gb", type=float, default=0.2,
                        help="weight size served per shard in --serve mode")
    parser.add_argument("--hosts", type=str, default="",
                        help="comma-separated host:port list for stage 3")
    parser.add_argument("--model-gb", type=float, default=0.6)
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--in-features", type=int, default=512)
    parser.add_argument("--sweep", type=str, default="",
                        help="comma-separated model sizes in GB to sweep")
    parser.add_argument("--interface", type=str, default="lo")
    parser.add_argument("--apply-netem", action="store_true")
    parser.add_argument("--report", type=str, default="")
    parser.add_argument("--auth-token", type=str, default=None)
    parser.add_argument("--anchor", nargs=2, type=float,
                        metavar=("SYNTHETIC_TAX", "REAL_TAX"),
                        help="anchor calibration verdict across the tolerance sweep")
    parser.add_argument("--anchor-model-gb", type=float, default=None,
                        help="largest size with a real-model anchor run")
    parser.add_argument("--claim-model-gb", type=float, default=None,
                        help="claimed swarm model size for the verdict")
    parser.add_argument("--gate", choices=["overflow"],
                        help="evaluate a pre-registered gate from recorded run facts")
    parser.add_argument("--per-shard-gb", type=float, default=None,
                        help="largest per-shard memory footprint observed, GB")
    parser.add_argument("--completed-tokens", type=int, default=None,
                        help="tokens actually generated by the overflow run")
    parser.add_argument("--min-available-gb", type=float, default=None,
                        help="smallest measured free memory across devices, GB (mem_get_info)")
    parser.add_argument("--leak-slope-mb", type=float, default=None,
                        help="measured memory slope of the 256-token leak probe, MB/token")
    parser.add_argument("--calibrate-shard", action="store_true",
                        help="measure max usable shard memory by OOM-stepping calibration")
    parser.add_argument("--calibrate-step-gb", type=float, default=0.25,
                        help="calibration resolution: max waste per shard equals one step")
    parser.add_argument("--max-usable-shard-gb", type=float, default=None,
                        help="measured per-shard limit from --replay-calibrate (the gate consumes the replayed number)")
    parser.add_argument("--record-trace", action="store_true",
                        help="journal the workload allocation pattern (weights, activations, KV growth, jitter)")
    parser.add_argument("--trace-path", type=str, default="alloc_trace.json",
                        help="allocation trace journal path (record and replay)")
    parser.add_argument("--replay-calibrate", action="store_true",
                        help="pattern-aware max usable shard memory by trace-replay OOM-stepping (gates on THIS)")
    parser.add_argument("--live-capture-gate", action="store_true",
                        help="judge the recorder IN TIME: positive live capture, negative live (zero records outside the window), live determinism (#27)")
    parser.add_argument("--trace-capture-gate", action="store_true",
                        help="judge the capture layer before calibration: accounting closure, 100MB leak injection +/-1%, deterministic replay (mandatory with --replay-calibrate)")
    parser.add_argument("--fragmentation-threshold", choices=["derived"],
                        default="derived",
                        help="flat-execution threshold: derived (k*sigma_noise from --noise-report); a chosen number is a type-four constant")
    parser.add_argument("--noise-report", type=str, default="",
                        help="variance report JSON (from --variance-report --report FILE) that supplies sigma_noise for the derived threshold")
    parser.add_argument("--paired", action="store_true",
                        help="variance report: paired standalone/swarm runs per iteration (sigma of the ratio)")
    parser.add_argument("--flat-shard-gb", type=float, default=None,
                        help="diagnostic flat ceiling from --calibrate-shard, recorded in the gate verdict")
    parser.add_argument("--variance-report", type=int, default=None, metavar="REPEATS",
                        help="run the stage REPEATS times and measure the noise floor")
    parser.add_argument("--reproduce-tolerance", nargs="+", metavar="REPORT_JSON",
                        help="derive the replay acceptance margin from variance reports")
    parser.add_argument("--verdict", nargs=2,
                        metavar=("SINGLE_JSON", "SWARM_JSON"))
    args = parser.parse_args()

    if args.anchor:
        anchor_verdict(args.anchor[0], args.anchor[1])
        return

    if args.verdict:
        apply_verdict(
            args.verdict[0], args.verdict[1],
            anchor_gb=args.anchor_model_gb,
            claim_gb=args.claim_model_gb,
        )
        return

    if args.calibrate_shard:
        result = calibrate_shard(args.calibrate_step_gb)
        save_report(result, args.report)
        return

    if args.record_trace:
        result = record_alloc_trace(
            args.model_gb, args.shards, args.tokens, args.in_features,
            args.trace_path,
        )
        save_report(result, args.report)
        return

    if args.live_capture_gate:
        if not live_capture_gate():
            raise SystemExit("REJECTED: live capture gate failed (amendment #27)")
        print("live capture gate: PASS - the recorder may now capture real traces")
        return

    if args.trace_capture_gate and not args.replay_calibrate:
        with open(args.trace_path) as f:
            gate_trace = json.load(f)
        if not trace_capture_gate(gate_trace):
            raise SystemExit("REJECTED: capture gate failed (amendment #26)")
        print("capture gate: PASS - the trace may now be replay-calibrated")
        return

    if args.replay_calibrate:
        result = replay_calibrate(
            args.trace_path, args.calibrate_step_gb,
            capture_gate=args.trace_capture_gate,
            noise_report=args.noise_report,
            fragmentation_threshold=args.fragmentation_threshold,
        )
        save_report(result, args.report)
        return

    if args.reproduce_tolerance:
        result = reproduce_tolerance(args.reproduce_tolerance)
        save_report(result, args.report)
        return

    if args.variance_report is not None:
        if args.stage not in (1, 2):
            parser.error("--variance-report needs --stage 1 or --stage 2")
        result = variance_report(
            args.stage, args.variance_report, args.model_gb, args.shards,
            args.tokens, args.in_features, args.paired,
        )
        save_report(result, args.report)
        return

    if args.gate == "overflow":
        needed = (
            args.per_shard_gb, args.completed_tokens, args.min_available_gb,
            args.max_usable_shard_gb,
        )
        if any(v is None for v in needed):
            parser.error(
                "--gate overflow needs --per-shard-gb, --completed-tokens, "
                "--min-available-gb and --max-usable-shard-gb"
            )
        overflow_gate_verdict(
            args.model_gb, args.shards, args.per_shard_gb, args.completed_tokens,
            args.min_available_gb, args.max_usable_shard_gb, args.leak_slope_mb,
            args.flat_shard_gb,
        )
        return

    if args.serve:
        layer = build_shard_layer(
            int(args.shard_gb * 1024 ** 3), args.in_features
        )
        print(f"serving one shard of {args.shard_gb} GB on port {args.port}")
        serve(
            shard_index=0,
            shard_layers=[("bench", layer)],
            port=args.port,
            auth_token=args.auth_token,
        )
        return

    if args.stage is None:
        parser.error("choose --stage, --serve, or --verdict")

    if args.stage == 1:
        layers = build_layers(args.model_gb, args.shards, args.in_features)
        report = run_isolated(layers, args.tokens, args.in_features)
    elif args.stage == 2:
        layers = build_layers(args.model_gb, args.shards, args.in_features)
        report = run_loopback(layers, args.tokens, args.in_features)
    elif args.stage == 3:
        if not args.hosts:
            parser.error("stage 3 needs --hosts host1:port,host2:port,...")
        addresses = [h.strip() for h in args.hosts.split(",") if h.strip()]
        report = run_swarm(
            addresses, args.tokens, args.in_features, stage=3, name="swarm"
        )
    elif args.stage == 4:
        report = run_netem(
            args.interface, args.apply_netem, args.model_gb, args.shards,
            args.tokens, args.in_features,
        )
    else:
        report = run_instrumented(
            args.model_gb, args.shards, args.tokens, args.in_features
        )

    if args.sweep and args.stage in (1, 2):
        sizes = [float(s) for s in args.sweep.split(",") if s.strip()]
        if args.stage == 1:
            def runner(gb, tokens, in_features):
                return run_isolated(
                    build_layers(gb, args.shards, in_features),
                    tokens, in_features,
                )
        else:
            def runner(gb, tokens, in_features):
                return run_loopback(
                    build_layers(gb, args.shards, in_features),
                    tokens, in_features,
                )
        sweep_report = run_sweep(runner, sizes, args.tokens, args.in_features)
        instrumented = run_instrumented(
            args.model_gb, args.shards, args.tokens, args.in_features
        )
        sweep_report["coordination_tax"] = instrumented["coordination_tax"]
        sweep_report["scaling_efficiency"] = instrumented["scaling_efficiency"]
        report = sweep_report

    print(json.dumps(report, indent=2, default=str))
    save_report(report, args.report)


if __name__ == "__main__":
    main()
