# RESEARCH_QUESTIONS.md - the open research frontier (issue #36)

This file is a frontier, not a commitment ledger. Everything
in it is a definition, an open question, or a registered
hypothesis; nothing here is a claim, and nothing here enters
README.md, CONTRIBUTING.md or PRE_REGISTERED.md as a
commitment until its definition freezes and its measurement
runs. The unified threshold law (CONTRIBUTING.md;
PRE_REGISTERED.md section 10) governs every threshold these
questions eventually produce.

## Model B (kappa)

Definition: Model B is the eviction-aware companion of the
token time law in MODELS.md. kappa is its measured
amplification coefficient - the factor by which page-miss
eviction amplifies the measured T_token beyond the
bandwidth-law bound computed over activated bytes:

    kappa = T_token(measured, with misses) /
            T_token(bandwidth law, activated bytes)

Open question: is kappa a device constant, a function of the
task class and length scale kappa(C, L), or a function of the
paging policy? Until it is measured, kappa is undefined as a
number, and Model B stays outside every threshold law.

## L1 (delta)

Open hypothesis: the latency term of the token time law is
constant as the length scale grows. Formally, with L0 the
campaign's reference length scale:

    delta(L) = latency(L) - latency(L0)

Hypothesis: delta(L) = 0 within the derived threshold of the
timing instrument, across the length scales the campaign
tests. Registered here before any measurement; refuted or
supported only by the campaign, never by intuition.

## I_eff - the candidate agents

I_eff is the effective agent set of a swarm run: the devices
that actually contribute inference, not the devices that are
merely present. Candidate definitions, to be decided by
measurement:

1. every device that passes the admission gate (the budget
   check);
2. every device with a measured throughput share above zero;
3. every device whose removal measurably degrades swarm TPS.

Open question: which of the three definitions predicts the
swarm advantage ratio of PRE_REGISTERED.md section 1? The
campaign decides; no definition is promoted before it wins on
data.

## h(C, L, policy, language) - the measurement constraint

h is the quality function of MODELS.md: task class C, length
scale L, decoding policy, natural language. The constraint on
every h measurement: h is measured on the configuration it
will judge, one variable held still at a time, and
h_threshold is derived from the measured cloud reference
T_cloud on the same task - never a fixed round number.

## H_lang - a registered hypothesis

Hypothesis: h varies across natural languages beyond the
derived noise threshold of its instrument - the quality
function is language-dependent, not language-invariant.
Registered here before any measurement. If the campaign
refutes it, h collapses toward a language-invariant form and
gets cheaper to measure; if the campaign supports it, every
language claim must carry its own measured h. No direction is
assumed.

## C_accept / C_reject - definition required

The cost ratio of the k derivation (MODELS.md, issue #36):
C_accept is the cost of accepting a false parity (a wrong
engine ships as verified); C_reject is the cost of rejecting a
true one (a correct engine is sent back). Both are
placeholders (ratio = 100) until they are defined and measured
in a separate issue. That issue must name whose costs they
are and in what units, before any k* computed from them
replaces the provisional k = 3.

## Status

No question in this file has been measured. The frontier
moves only by campaign data, under the amendment rule of
PRE_REGISTERED.md section 11.
