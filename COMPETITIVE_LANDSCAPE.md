# Competitive Landscape (wave #48)

The HiveMind arena is interactive, single-user, LOCAL inference on
constrained hardware: 8-16 GB RAM, no GPU, no server. This map
positions the project against the systems it gets compared to, and
names the blue water where none of them sail.

## The arena is not the server rack

| System | Where it fights | What it optimizes | Why it is not HiveMind |
|---|---|---|---|
| FlexGen | Server GPUs, batch serving | Throughput: IO-efficient offloading, compression, LARGE effective batch sizes | Trades latency for throughput; built for offline, throughput-oriented batch workloads on commodity server GPUs, not interactive desktop use |
| DeepSpeed | Clusters and servers (training + inference) | ZeRO sharding, memory-efficient optimizers, multi-GPU scale | A training and optimization stack for fleet-scale workloads; assumes someone operates servers |
| llama.cpp | Local inference (closest neighbor) | Quantized CPU/GPU inference, mmap-backed weights | The closest - but it hands memory policy to the OS pager (below) |

## The mmap trap (llama.cpp)

llama.cpp maps model weights with the OS pager: pages arrive on demand
and the OS decides eviction. When the model exceeds available RAM,
the OS pages out hot pages in the middle of generation - the llama.cpp
documentation itself warns that under memory pressure mmap "might
increase the risk of pageouts, negatively impacting performance". On
a 16 GB machine the pager is also fighting the browser, the IDE and
the OS: silent latency spikes, no declared budget, no accountability.

## The HiveMind blue water

1. **Explicit budget covenant.** HiveMind owns its memory policy:
   the budget is declared (LRUCache page limits plus MemoryMonitor),
   a page loads only if it fits, and eviction is first-class and
   predictable. The OS pager never surprises the user mid-sentence.
2. **The easy surface.** One Setup.exe, no admin rights, a
   streamlit UI, a tray icon, a silent one-time core download.
   Where the neighbor says "just compile it", HiveMind says "just
   click it".
3. **Router as Oracle (expert-granularity paging).** Dense engines
   must page whole layer stacks; the HiveMind router knows which
   experts the next tokens need and can prefetch only those pages
   (design: docs/ROUTER_AS_ORACLE.md). Neither OS mmap nor batch
   engines can offer expert-level memory control.

## What HiveMind does NOT claim

- Not faster than llama.cpp at equal quantization on a dense model -
  the neighbor has years of kernel-level tuning.
- Not a server: no fleet batching; one interactive stream at a time.
- The blue water is the COMBINATION - budgeted, easy, and
  router-aware - not any single feature.

Verified against the FlexGen paper pages and the llama.cpp
documentation, wave #48.
