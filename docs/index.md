# HiveMind

**Run 7B+ LLMs on low-end devices via Neural Swarm Inference.**

HiveMind is an out-of-core LLM inference engine: it runs models far larger
than your RAM by paging model weights from disk, quantizing them down to
2 bits, and distributing work across a swarm of machines.

## Highlights

- **Out-of-core inference** — page weights in and out of RAM with an
  adaptive page size (up to 4000 MB pages for 50B+ models).
- **Q2 / Q4 / Q8 quantization** — 2-bit quantization packs four values per
  byte and unlocks 50B-100B models on a single 32 GB machine.
- **Large model presets** — reference configurations for 50B, 70B and 100B
  models with GQA attention (`TransformerConfig.from_preset`).
- **Neural Swarm** — distributed inference over gRPC: several low-end
  devices acting as one bigger machine.
- **Memory-aware** — a dynamic quantizer picks the highest bit-width that
  fits the available RAM.

## Documentation

- [Installation](installation.md)
- [User Guide](user_guide.md)
- [API Reference](api_reference.md)
- [Contributing](contributing.md)
