# Journal 01 · First replication experiment

**Date:** 2026-09-24
**Notebooks:** `01_exploration.ipynb`, `02_replication_clustering.ipynb`

## What I set out to do

Validate the pipeline before trying to discover anything new: can unsupervised clustering recover the expert coda types of the EC1 clan without seeing the labels?

## Decisions and why

- **Data:** DSWP coda dataset (Sharma et al. 2024), 8,719 codas. Kept EC1 only, removed `*-NOISE`, kept codas with ≥ 3 clicks → 7,268 codas, 24 types.
- **Timing, not spectrograms:** codas carry information mainly in inter-click intervals, so features are ICI-based. This changes the original plan (spectral features first); spectral analysis moves to a later phase.
- **Cluster within each click count:** click count is directly observable, so there is no point asking the algorithm to rediscover it. It also avoids zero-padding distorting distances.
- **k chosen by BIC**, never by the labels.
- **Four metrics, not one:** ARI alone would have hidden the most interesting result. Homogeneity and completeness separate "splits a type" from "merges types".

## Things I found in the data

- Two clans are mixed in the file (EC1: 7,770, EC2: 949). Some types are clan markers.
- About 2/3 of codas have no identified whale (`IDN = 0`). Individual-level analyses will use a much smaller subset.
- Dates mix `/` and `-` separators; all are day-first (verified).
- Strong imbalance: `1+1+3` alone is ~41% of the data.
- EC1 contains 24 non-noise types rather than the 21 usually reported, because a few "EC2-style" types appear in small numbers.

## Results

| features | ARI | NMI | homogeneity | completeness |
|---|---|---|---|---|
| rhythm | 0.43 | 0.71 | 0.84 | 0.61 |
| rhythm + tempo | 0.49 | 0.79 | 0.96 | 0.68 |
| raw ICIs | 0.48 | 0.79 | 0.95 | 0.68 |

- Tempo matters: without it, `5R1`, `5R2`, `5R3` (same regular shape, different speeds) cannot be separated.
- The clusters are finer than the expert types. The main case: **`1+1+3` splits by tempo.**
- Not an artefact of recording context: every social unit uses every tempo sub-group; NMI with unit and with year ≈ 0.05. Units do show preferences (e.g. unit U mostly slow, unit A mostly mid).

## What didn't work (and what I learned)

The full rhythm+tempo model split `1+1+3` consistently, but **where** it drew the boundaries changed between random seeds (partition ARI 0.53–0.61). On that evidence alone I could not claim tempo classes.

A focused test fixed this: a 1-D GMM on log(duration) of `1+1+3` only selects **k = 3 in 5/5 seeds and 30/30 bootstrap resamples**, with centres at **0.81 s, 1.05 s and 1.28 s** and identical partitions across seeds. ΔBIC: +109 for k = 2, +33 for k = 4.

Lesson: the broad model is useful for *noticing* structure; confirming it needs a targeted test.

## Open questions

1. Do other rhythm types share the same tempo centres? (Sharma et al. describe tempo as a small set of shared classes.)
2. `5R1` splits into three clusters with the *same* duration. Real fine-rhythm variation, or the GMM carving a continuous cloud?
3. Do social units differ significantly in tempo usage once non-independence is handled?
4. Does the whole pipeline hold on EC2?

## Next session

Start with question 1: apply the 1-D tempo test to every rhythm type with enough data and compare the centres.
