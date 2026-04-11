# DSAI_ML_Project
# ACWA Power — Alkaline Electrolyzer Fault Detection & Risk Classification

> **Machine Learning project | Tuwaiq Academy **  
> Multi-class classification on unlabeled operational sensor data from an Alkaline Water Electrolyzer (AWE)

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Approach Overview](#approach-overview)
5. [Approach 1 — Clustering-Based Labeling (V1)](#approach-1--clustering-based-labeling-v1)
6. [Approach 2 — Threshold-Based Labeling (V2) ✅ Best Results](#approach-2--threshold-based-labeling-v2--best-results)
7. [Models Used](#models-used)
8. [EDA Summary](#eda-summary)
9. [Feature Engineering](#feature-engineering)
10. [Evaluation Strategy](#evaluation-strategy)
11. [Requirements](#requirements)

---

## Problem Statement

Alkaline Water Electrolyzers (AWE) are critical assets in green hydrogen production. They operate continuously under harsh electrochemical conditions and are subject to a range of failure modes including:

- **Gas crossover** (H₂ into O₂ side or vice versa) — diaphragm degradation
- **Electrode passivation** — surface oxidation and scaling
- **Electrolyte leakage / contamination** — gasket failures, thermal cycles
- **Corrosion** — high KOH concentration and temperature
- **Diaphragm rupture** — high differential pressure in H₂/O₂ separators
- **Water quality degradation** — DM water conductivity exceeding safe thresholds

The dataset contains **no ground-truth labels**. The goal is to build a multi-class classification system that predicts the **operational risk level / maintenance need** of the electrolyzer at each timestep, using only sensor readings.

---

## Dataset

| Property | Detail |
|---|---|
| Source | ACWA Power — Industrial Hackathon dataset (`ACWA_DATASET.csv`) |
| Type | Time-series sensor readings from a single AWE stack |
| Labels | **None** (fully unsupervised at input) |
| Preprocessing | Duplicates removed; `Time` and `Temperature_WTP_production` dropped |
| Idle detection | `is_idle = 1` where `Voltage_1_Stack == 0` |

### Key Sensor Features

**Electrical:** `Current_1_stack`, `Voltage_1_Stack`, `DC_Power_Consumption_1_Stack`

**Thermal:** `Room_temperature`, `H2_side_outlet_temp_1_stack`, `O2_side_outlet_temp_1_stack`, `Lye_Supply_to_Electrolyzer_Temp`

**Lye & Separator:** `Lye_Concentration`, `Lye_Flow_to_1_Stack`, `H2_Separator_Level`, `O2_Separator_Level`, `LDI_H2_&_O2_Separator`

**Gas Purity & Safety:** `O2_content_in_H2`, `H2_content_in_O2`, `Pressure_O2_Separator`

**Flow & Water Quality:** `H2_Flowrate_Purification_outlet`, `DM_water_condctivity`, `DM_water_flow_from_B.L.`

---

## Project Structure

```
├── final_preprossing_EDA_AI_V1.ipynb   # Approach 1: Clustering-based labeling
├── final_preprossing_EDA_AI_V2.ipynb   # Approach 2: Threshold-based labeling (best)
├── Acwa_power.pdf                      # Supporting document — AWE failure modes
├── ACWA_DATASET.csv                    # Raw sensor data 
└── README.md
```

---

## Approach Overview

Since no labels exist, both approaches follow a **two-stage pipeline**:

```
Raw Sensor Data
      │
      ▼
  EDA & Preprocessing
      │
      ├──► Approach 1: K-Means Clustering → pseudo-labels → Classification
      │
      └──► Approach 2: Domain-Threshold Rules → binary labels → Classification ✅
```

Both approaches use **the same set of classification algorithms**, evaluated with the same training and cross-validation strategies, allowing fair comparison.

---

## Approach 1 — Clustering-Based Labeling (V1)

**Notebook:** `final_preprossing_EDA_AI_V1.ipynb`

### Label Generation

1. **Scaling** — `RobustScaler` applied to all numeric features (robust to outliers from fault events)
2. **Dimensionality Reduction** — PCA to 2 components on a curated subset of risk-relevant features
3. **Optimal k selection** — Elbow method, Silhouette score, and Davies-Bouldin index evaluated for k = 2 to 7 → **k = 4** selected
4. **Cluster-to-severity mapping** — Clusters ranked by domain-informed gas risk proxy score:
   - `gas_risk_score = mean(O2_content_in_H2) + mean(H2_content_in_O2)` per cluster
   - Clusters sorted from lowest to highest → mapped to severity labels `0, 1, 2, 3`

### Risk Level Mapping

| Label | Severity | Description |
|---|---|---|
| 0 | Normal | Low gas crossover, stable electrolyte |
| 1 | Low Risk | Slightly elevated indicators |
| 2 | Medium Risk | Notable anomalies in purity or pressure |
| 3 | High Risk | Critical gas crossover or separator imbalance |

### Limitations

- Cluster boundaries are data-driven, not physics-driven — may not align with actual failure thresholds
- Class imbalance handled with **SMOTE** in some training variants
- PCA projection may merge physically distinct fault modes that appear similar in reduced space

---

## Approach 2 — Threshold-Based Labeling (V2) ✅ Best Results

**Notebook:** `final_preprossing_EDA_AI_V2.ipynb`

### Label Generation

Labels are assigned using **domain-knowledge thresholds** derived from AWE safety standards and the ACWA Power supporting document:

```python
def label_maintenance(row):
    # 0. Idle — no assessment
    if row['Voltage_1_Stack'] == 0:
        return 0

    # 1. Critical gas purity violations
    if row['O2_content_in_H2'] > 1:        # safety limit: 0.5%
        return 1
    if row['H2_content_in_O2'] > 2:        # safety limit: 1.5%
        return 1
    if row['LDI_H2_&_O2_Separator'] > 2:   # separator imbalance
        return 1

    # 2. Electrolyte concentration out of range (normal: 25–32 wt%)
    if row['Lye_Concentration'] < 25 or row['Lye_Concentration'] > 32:
        return 1

    # 3. Water quality degradation
    if row['DM_water_condctivity'] > 2:    # ideal: < 1 µS/cm
        return 1

    return 0  # Normal operation
```

**Target variable:** `Maintenance_Needed` (binary: 0 = Normal, 1 = Maintenance Required)

### Why This Approach Gives Better Results

- Labels are grounded in **physical failure thresholds**, not statistical clusters
- Eliminates the risk of cluster misalignment with actual fault boundaries
- Produces a cleaner signal for downstream classifiers
- Binary framing is more actionable for real maintenance scheduling

---

## Models Used

Both approaches train and evaluate the **same model suite**:

| Model | Notes |
|---|---|
| Gaussian Naive Bayes | Baseline probabilistic model |
| Bernoulli Naive Bayes | V2 only |
| Logistic Regression | V2 only |
| SVC (RBF kernel) | Support Vector Classifier |
| KNN | k=4 (V1), k=2 (V2) |
| Random Forest | 200 estimators |
| Gradient Boosting | 200 estimators |
| Extra Trees | 200 estimators |
| AdaBoost | 200 estimators |
| Bagging Classifier | 200 estimators |

---

## EDA Summary

### Idle vs Active
The dataset contains a significant proportion of idle readings (`Voltage_1_Stack == 0`). During idle, all process features collapse to zero or ambient values. These are excluded from risk labeling in both approaches.

### Electrical Features
Current, Voltage, and DC Power are tightly correlated (r ≥ 0.95). During active operation, the stack runs near full load (~7,800–8,000 A), with bimodal distributions driven by idle/active separation.

### Thermal Features
Room temperature is stable (24–26 °C). H₂ and O₂ outlet temperatures reach 80–90 °C during active operation, which is the expected AWE operating range.

### Gas Purity (Critical Safety Indicators)
- **O₂ in H₂** safety limit: **0.5%** — a notable fraction of active readings exceed this
- **H₂ in O₂** safety limit: **1.5%** — crossover events visible in the tail of the distribution
- These features are **excluded from model input** (used only for labeling) to prevent data leakage

### Lye & Separator
- `LDI_H2_&_O2_Separator` > 1.5 signals dangerous gas mixing risk
- Lye concentration normal range: 25–32 wt%

### Water Quality
- DM water conductivity should stay below 1 µS/cm; readings above this indicate contamination risk

---

## Feature Engineering

Features used in the **model input** (after removing label-leaking columns):

```python
MODEL_FEATURES = [
    "Room_temperature",
    "Current_1_stack",
    "DC_Power_Consumption_1_Stack",
    "H2_side_outlet_temp_1_stack",
    "O2_side_outlet_temp_1_stack",
    "Lye_Supply_to_Electrolyzer_Temp",
    "Lye_Flow_to_1_Stack",
    "Pressure_O2_Separator",
    "H2_Flowrate_Purification_outlet",
    "DM_water_flow_from_B.L.",
]
```

> Features that directly encode the label (`O2_content_in_H2`, `H2_content_in_O2`, `Voltage_1_Stack`, `LDI_H2_&_O2_Separator`, `Lye_Concentration`, `DM_water_condctivity`) are **dropped from X** to prevent leakage.

---

## Evaluation Strategy

Four evaluation protocols are used across both approaches, enabling robust comparison:

| Protocol | Description |
|---|---|
| **Train/Test Split** | 80/20 stratified split, no augmentation |
| **Train/Test + SMOTE** | SMOTE oversampling applied inside the training fold only |
| **Stratified K-Fold CV** | 5-fold CV, no SMOTE |
| **Stratified K-Fold CV + SMOTE** | SMOTE applied per fold inside an `ImbPipeline` |

**Metrics reported:** Accuracy, Weighted F1, Weighted Recall, Precision, Confusion Matrix

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

---

## Key Takeaways

- **Threshold-based labeling (V2) outperforms clustering-based labeling (V1)** because domain knowledge provides more precise and physically meaningful decision boundaries than unsupervised cluster centroids.
- Tree-based ensemble methods (Random Forest, Extra Trees, Gradient Boosting) consistently outperform simpler models due to the non-linear, feature-interaction-heavy nature of electrolyzer fault patterns.
- **Gas purity and separator differential** are the most discriminative risk indicators but must be excluded from model input to avoid leakage.
- SMOTE improves recall on the minority fault class but should be applied strictly inside cross-validation folds to avoid optimistic bias.
