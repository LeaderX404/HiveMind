# Router as Oracle: feeding the prefetch thread (wave #48 design doc)

Status: DESIGN, ready for review. Nothing in the runtime changes
until each phase is implemented behind its config flag.

## 1. Problem

PagedTransformerEngine (hivemind/core/transformer_engine.py) runs a
strictly sequential forward pass: layer i computes only after layer
i-1 returned. Every weight load is a synchronous _load_tensor() ->
file_pager.load_tensor() call; a cache miss stalls the whole
generation on disk I/O. The engine pays latency for every cold page
even though the sequence of future tensor loads is predictable.

Two prediction signals exist:

- a) DENSE LOOKAHEAD: the layer stack order is known. While layer i
  computes (CPU-bound matmuls), layer i+1 tensors can be read from
  disk in parallel (I/O-bound) - classic compute/IO overlap.
- b) ROUTER ORACLE: in sparse/MoE models the router chooses which
  expert weights the next layers need. The router decision IS the
  prefetch list: load only the chosen experts, skip the rest -
  expert-granularity paging that dense engines and OS mmap cannot
  do.

## 2. Design constraints (non-negotiable)

- C1 ZERO BREAK: with prefetch disabled (default), behavior must be
  identical to today - engine, pager and tests untouched.
- C2 BUDGET COVENANT: prefetching never breaches the declared RAM
  budget. Every candidate page passes MemoryMonitor.can_load_page;
  prefetched pages count against the same LRUCache capacity; on
  pressure, prefetched (cold) pages evict first.
- C3 NO DATA RACES: tensor_cache and file_pager become shared
  state. All loads (main thread + prefetch thread) serialize behind
  one lock; the hot path takes no lock it does not take today when
  prefetching is off.
- C4 BEST-EFFORT PREDICTIONS: a wrong or late prediction is a
  miss, never an error. The queue is bounded; overflow drops the
  oldest predictions (backpressure by discard).
- C5 MEASURABLE: hit / miss / late / discard counters on every
  phase, exported through engine.get_stats().

## 3. Architecture

```
+--------------------+     notify(name)     +---------------------+
| PagedTransformer  |--------------------->| TensorPrefetcher     |
|  _load_tensor()    |                     |  (daemon thread)     |
+--------------------+                     |  bounded deque of    |
        ^                                  |  predicted names     |
        |  predictions load through the SAME choke point,        |
        |  under the prefetch lock: engine._load_tensor()         |
        +---------------------------------------------------------+
```

- TensorPrefetcher: new module hivemind/core/prefetch.py. Owns one
  daemon thread and one bounded collections.deque (maxlen from
  config).
- Engine integration (the ONLY engine change, C1): in __init__,
  self.prefetcher = None; in _load_tensor(), after a load:
      if self.prefetcher is not None:
          self.prefetcher.notify(tensor_name)
  Nothing else in transformer_engine.py moves.
- Locking (C3): a single engine-level RLock, created lazily ONLY
  when a prefetcher is attached. _load_tensor wraps its body in it
  when present; without a prefetcher the code path is unchanged.
- The prefetch thread runs predictions through the SAME
  engine._load_tensor() - caches, stats and budget checks stay
  coherent (single source of truth).

## 4. The router contract (Phase B)

```
class RouterListener(Protocol):
    def on_decision(self, layer_index: int,
                    expert_ids: tuple,
                    confidence: float) -> None: ...
```

- The router emits AFTER computing its decision, BEFORE the expert
  layers run - that interval is the prefetch window.
- The prefetcher maps decisions to page names via the
  layer_to_page / expert-to-page tables and enqueues them.
- Dense lookahead (Phase A) is the degenerate policy: no router
  needed; notify(layer i tensors) enqueues layer i+1.

## 5. Phases

- Phase 0 - SHADOW: counters only, no thread, no behavior.
- Phase 1 - DENSE LOOKAHEAD: prefetch layer i+1 for dense models.
  Config: engine.prefetch_mode = "off" | "dense" (default "off").
- Phase 2 - ROUTER ORACLE: expert-granularity for MoE configs.
- Phase 3 - TUNING: prefetch depth scaled by confidence; adaptive
  budget share (e.g. at most 25 percent of the page cache for
  prefetched pages).

## 6. Failure modes and answers

- THRASH: a prefetched page evicts a hot one - mark prefetched
  pages cold in the LRU so they evict first.
- DISK CONTENTION: prefetch I/O slows the main miss path - the
  single lock serializes loads; C5 metrics will show whether the
  main thread ever waits on the prefetcher.
- BUDGET BREACH: the can_load_page gate before every prefetch.
- WRONG PREDICTIONS: bounded deque drops them; persistent low
  hit-rate flags the config to disable per model.

## 7. Test plan

- Unit: fake FileBackedPager; deterministic scripts; assert
  identical logits with prefetch off vs on (C1).
- Race: stress test, 10k loads from two threads; no exceptions,
  no duplicate loads under the lock.
- Budget: tiny fake RAM; assert prefetched pages evict first and
  no budget breach (C2).
- Benchmark: A/B latency, page-cache hit rate, tokens per second
  on the 7B preset, cold and warm.

## 8. Open questions for review

- Q1: Is prefetch depth above 1 worth it at 1 GB pages?
- Q2: Should the KV cache growth policy share the prefetch budget
  (same RAM pool)?
- Q3: Is one daemon thread enough at Phase 2, or per-expert I/O?
