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
