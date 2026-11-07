# Limitations

Honest statements about what HiveMind cannot do today. Nothing in
this file is marketing. Derived facts are marked DERIVED
(arithmetic from named assumptions); anything not yet measured is
explicitly called unmeasured. The protocol that will replace
"unmeasured" with numbers is pre-registered in PRE_REGISTERED.md.

## 70B-class models: seconds per token (DERIVED)

A dense 70B model must touch all of its weights every token. At Q2
that is ~18GB of weights streaming per token (parameters x bits
per parameter, matching the memory table in the README). At an
NVMe stream rate of ~5GB/s the best case is ~3.6 seconds per
token.

Seconds per token is batch/offline processing, not conversation.
If you need interactive chat, this engine - today, on a single
limited-RAM device - is not the tool for a 70B model.

## What is measured, what is derived, what is untested

- Measured on hardware: nothing yet. The 3-device measurement
  campaign is pending; its protocol is pre-registered in
  PRE_REGISTERED.md, and measured numbers will carry their margin
  as an interval, never a bare point.
- Derived (arithmetic, not performance claims): memory requirements
  per quantization level; the seconds-per-token floor above.
- Untested: the 50B-100B presets (see README, Experimental).

## Scope

- The out-of-core target is a single device with limited RAM and
  NVMe-class storage. Nothing here is tested for networked
  storage.
- Quantization formats: Q2, Q4, Q8. Anything else is untested.

This file grows when a new limitation is discovered - and shrinks
only when a measurement, recorded under the covenant's protocol,
says it may.
