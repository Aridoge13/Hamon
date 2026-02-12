# HAMON

Health Adaptive Monitoring and Optimization Network

HAMON is a cardiometabolic risk interpretation framework built on longitudinal wearable data, manual health inputs, temporal machine learning, and structured clinical language model reasoning.

The system is designed for research and educational purposes. It does not perform medical diagnosis.

---

## Overview

HAMON analyzes at least 30 days of personal health data to learn individualized baselines. It detects sustained deviations across cardiovascular and metabolic signals and classifies users into interpretable risk states. A clinical language model then translates structured model outputs into cautious, human readable explanations.

The project focuses on architectural rigor, uncertainty modeling, and safety constraints.

---

## Problem Statement

Wearable health data is often interpreted using short term averages or generic thresholds. This approach ignores personal baselines, measurement gaps, and cross system interactions.

HAMON addresses this by:

* Learning personalized baselines
* Modeling temporal deviations rather than single day snapshots
* Incorporating missingness as an informative feature
* Producing calibrated risk probabilities
* Separating prediction from language based interpretation

---

## Data Modalities

### Wearable Signals

* Resting heart rate
* Heart rate variability
* Activity intensity minutes
* Step count
* Sleep duration if available
* VO2max proxy if available

### Manual Inputs

* Fasting glucose
* Post prandial glucose
* Blood pressure
* SpO2
* Optional menstrual phase modifier

Synthetic data is generated using physiologically coherent latent variables to ensure realistic correlations between signals.

---

## Core Architecture

### 1. Baseline Engine

For each individual the system computes:

* 30 day baseline median
* Rolling 7 day averages
* Exponential moving trends
* Percent deviation from baseline
* Missingness indicators

### 2. Risk Classifier

A LightGBM classifier predicts probability distributions across five cardiometabolic states:

1. Cardiometabolic Stable
2. Autonomic Stress Dominant
3. Emerging Metabolic Dysregulation
4. Elevated Cardiovascular Load
5. High Multisystem Risk with medical review advised

Probabilities are calibrated and accompanied by a confidence score that accounts for missing data.

### 3. Clinical Interpretation Layer

Med Gemma receives structured outputs including predicted class, key deviations, and uncertainty flags. The language model generates:

* Physiological explanation
* Conservative exercise guidance
* Explicit safety notes
* Clear referral triggers

The language model does not assign risk categories.

---

## Safety Design

HAMON includes strict guardrails:

* No disease diagnosis
* No treatment claims
* Hard constraints preventing high intensity recommendations under high risk states
* Explicit uncertainty communication

---

## Evaluation

Model evaluation includes:

* Stratified validation on synthetic cohorts
* Confusion matrices
* Calibration curves
* Sensitivity analysis for missingness

The project emphasizes interpretability and transparency over leaderboard optimization.

---

## Limitations

* Fully synthetic dataset
* No ECG integration
* No clinical outcome validation
* Wearable measurement bias

HAMON should be considered a research prototype.

---

## Repository Structure

```

```

---

## Strategic Goal

HAMON demonstrates disciplined medical AI system design under time constraints. The project highlights temporal modeling, uncertainty handling, and responsible integration of a clinical language model.
