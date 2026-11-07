# Measurement policy (UI)

The UI displays, it does not judge. These rules keep its numbers honest:

1. Every predicted number is displayed as a RANGE plus a SOURCE - never a bare point.
2. No number without its source; no source without its number.
3. `OVERHEAD_FACTOR = 0.7` is an internal detail of the calculator and never appears in any output.
4. The disk probe is an UPPER BOUND: it measures the disk on its best behavior, cache layers included.
5. The probe is graded by model size (`probe_size_for_model`): under 2 GB -> 100 MiB, 2-10 GB -> 500 MiB, 10-50 GB -> 2 GiB, above 50 GB -> 5 GiB.
6. A real load is journaled by `record_actual_load` right after the first run of a model; a measurement outranks every estimate.
7. The gap between the probe's upper bound and the measured sustained rate is the sustained-rate premium; it is recorded under `measured_bw` in `profile.json` and displayed next to the probe speed.
8. A measurement is shown as a band (roughly +/-10%), because a single load is one sample, not a law.
