# Discovering Hidden Structures in Sperm Whale Vocalizations Using Unsupervised Machine Learning

## Introduction

This project is a personal research initiative that explores whether modern Data Science and Machine Learning techniques can uncover hidden structures and recurring patterns within sperm whale vocalizations.

The idea was inspired by the documentary *Fathom: Decoding the Deep*, which follows researchers investigating the communication systems of sperm whales and the possibility that these animals exchange information in ways that are still not fully understood by humans.

After watching the documentary, I became interested in the intersection between marine science, artificial intelligence, signal processing, and data analysis. As a final-year Data Science and Engineering student, I saw an opportunity to explore this field from a data-driven perspective and document the entire research process in a public repository.

---

## Research Question

**Can unsupervised machine learning techniques reveal meaningful structures and patterns in sperm whale vocalizations?**

Rather than attempting to prove the existence of a language, this project focuses on a more measurable objective: identifying whether acoustic recordings contain naturally emerging groups, similarities, or recurring structures that can be detected through computational methods.

---

## Project Goals

The main goal of this research is to investigate sperm whale vocalizations using Data Science methodologies and explore whether hidden acoustic patterns can be discovered without relying on predefined labels.

The project aims to:

* Explore publicly available sperm whale acoustic datasets.
* Study the characteristics of underwater vocalizations.
* Extract meaningful acoustic features from recordings.
* Apply dimensionality reduction techniques to visualize patterns.
* Use clustering algorithms to identify naturally occurring groups.
* Analyze and interpret the discovered structures.
* Document every stage of the research process in a transparent and reproducible way.

---

## Why This Topic?

Whale communication remains one of the most fascinating open questions in modern biology.

Recent advances in machine learning, bioacoustics, and large-scale data analysis have created new opportunities to study complex animal communication systems. While many projects focus on classification tasks, this research is centered on discovery: searching for patterns that may not be immediately visible through traditional analysis.

This makes the project an opportunity not only to apply Data Science techniques but also to explore how computational methods can contribute to scientific research.

---

## Methodology

The project will progressively evolve through several stages:

### Data Collection

Compilation and evaluation of publicly available sperm whale vocalization datasets.

### Audio Processing

Preparation and transformation of raw acoustic recordings into machine-learning-ready representations.

### Feature Extraction

Extraction of acoustic characteristics such as frequency, energy, temporal patterns, and spectral descriptors.

Sperm whale codas carry most of their information in click **timing**, so the first phase works with inter-click intervals (ICIs), split into *rhythm* (ICIs normalised by coda duration) and *tempo* (coda duration). Spectral descriptors of individual clicks are planned as a second layer once audio is incorporated.

### Exploratory Analysis

Statistical and visual exploration of the extracted features.

### Dimensionality Reduction

Application of techniques such as PCA, t-SNE, and UMAP to reveal hidden structures within the data.

### Unsupervised Learning

Use of clustering algorithms to investigate whether naturally occurring acoustic groups emerge from the recordings.

### Interpretation

Analysis of results, limitations, and potential implications for future research.

---

## Current Status

**Phase 1 — Replication (in progress).** Before searching for new structure, the pipeline is validated by testing whether unsupervised clustering recovers the coda types already defined by experts for the EC1 clan of Dominica.

First results (see [`journal/`](journal/) and [`notebooks/02_replication_clustering.ipynb`](notebooks/02_replication_clustering.ipynb)):

- Clustering on rhythm + tempo within each click count produces almost pure clusters (homogeneity ≈ 0.96) that are finer than the expert types (completeness ≈ 0.68).
- The most frequent expert type, `1+1+3`, splits into **three tempo classes** (~0.81 s, 1.05 s, 1.28 s). The split is robust (k = 3 in 5/5 seeds and 30/30 bootstrap resamples) and is not explained by social unit or recording year. This is consistent with the tempo feature described by Sharma et al. (2024).
- Limitation: in the full multi-dimensional model, cluster boundaries are only moderately stable across random seeds.

---

## Repository Purpose

This repository serves as:

* A public research journal.
* A technical portfolio project.
* A learning platform for bioacoustics and machine learning.
* A record of the complete research process, including both successful and unsuccessful experiments.

The intention is to document the evolution of the project from the initial idea to the final conclusions, making the entire journey transparent.

---

## Repository Structure

```
whale-vocalization-patterns/
├── data/
│   ├── raw/            # downloaded data (not versioned)
│   └── processed/      # derived data (not versioned)
├── journal/            # dated research journal entries
├── notebooks/
│   ├── 01_exploration.ipynb
│   └── 02_replication_clustering.ipynb
├── reports/figures/    # figures generated by the notebooks
├── scripts/
│   └── download_data.py
├── src/whalecodas/     # reusable code (loading, features, clustering)
├── pyproject.toml
└── requirements.txt
```

## Getting Started

```bash
git clone https://github.com/amaisuarez/whale-vocalization-patterns.git
cd whale-vocalization-patterns
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                    # makes `import whalecodas` work everywhere
python scripts/download_data.py     # downloads the DSWP coda dataset
jupyter lab                         # open notebooks/ in order
```

## Data and Citation

The analysis uses the annotated coda dataset of the **Dominica Sperm Whale Project** (8,719 codas, 2005–2016), released under CC BY 4.0 with:

> Sharma, P., Gero, S., Payne, R., Gruber, D. F., Rus, D., Torralba, A., & Andreas, J. (2024). Contextual and combinatorial structure in sperm whale vocalisations. *Nature Communications*, 15, 3617. https://doi.org/10.1038/s41467-024-47221-8

Dataset and code: https://github.com/pratyushasharma/sw-combinatoriality · Archive: https://doi.org/10.5281/zenodo.10817697

The data is not redistributed in this repository; `scripts/download_data.py` fetches it from the original source at a pinned commit.
