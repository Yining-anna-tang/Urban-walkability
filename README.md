# Urban walkability inadequacy exacerbates emotion-induced distraction under the challenge of e-bikes expansion

This repository contains the data processing scripts, modeling code, and supplementary materials for the study:

**Urban walkability inadequacy exacerbates emotion-induced distraction under the challenge of e-bikes expansion**

---

## Overview

The rapid expansion of electric bicycles (e-bikes), especially in urban food-delivery systems, is reshaping pedestrian environments. While previous studies have mainly focused on physical safety risks such as traffic accidents and lithium battery explosions, the psychological and cognitive impacts on pedestrians remain underexplored.

This project investigates how e-bike encroachment into pedestrian spaces affects **emotion-induced distraction (EID)** and mental well-being, with a particular focus on **gender disparities** in urban mobility.

---

## Abstract

Urban walking experiences are increasingly challenged by the rapid expansion of electric bicycles (e-bikes). Existing research has focused primarily on fatalities and severe injuries caused by battery explosions and collisions. However, the psychological health risks faced by pedestrians under the ongoing “new-energy transport revolution” have been largely overlooked.

Using a nationwide social survey dataset (**N = 8,148**) and a supplementary controlled experiment (**N = 1,857**), this study applies a **Categorical Boosting (CatBoost)** model to provide prospective evidence that higher levels of **emotion-induced distraction (EID)** are significantly associated with urban e-bike encroachment (**R² = 0.601, p < 0.001**).

Gender-stratified analyses reveal that women exhibit higher susceptibility to EID, likely due to disproportionate unpaid caregiving responsibilities and greater exposure to mobility-related stress. These findings highlight the need to integrate pedestrian mental health considerations into urban transport planning and to address gender inequalities in mobility governance.

---

## Keywords

- Electric bicycles (e-bikes) expansion  
- Pedestrian mental health  
- Walking anxiety  
- Gender inequality  
- Urban public health  

---

## Data Sources

### 1. Chinese General Social Survey (CGSS 2021)

The primary dataset is derived from the **Chinese General Social Survey (CGSS) 2021**, which provides nationally representative data on:

- social governance and public services
- environmental attitudes and behaviors
- demographic and socio-economic characteristics

Only respondents who completed **Module F** were included, resulting in **8,148 valid observations** used in the main analysis.

### Key Variables

**Dependent variable**

- **Emotion-Induced Distraction (EID)**  
  Derived from survey item **E7**:  
  *"Due to emotional issues, your work or other daily activities have become absent-minded?"*

**Core explanatory variable**

- **Urban Walking Habit Frequency (UWHF)**  
  Derived from survey item **E20**:  
  *"How long do you walk on weekdays?"*  
  Converted into minutes per day.

---

### 2. Supplementary Experimental Dataset

To address the lack of specific e-bike exposure information in CGSS, we conducted an online controlled experiment.

- **Sample size:** 1,857 participants  
- **Platform:** https://www.credamo.com  
- **Design:** **2 × 2 between-subjects experiment**  
- **Target population:** Urban residents frequently exposed to e-bikes during short-distance walking

---

## Methods

### Machine Learning Model

We employ **CatBoost**, a gradient boosting algorithm optimized for categorical features, to model the relationship between:

- urban mobility conditions  
- demographic characteristics  
- psychological outcomes (EID)

**Model performance:**

```
R² = 0.601
p < 0.001
```

---

### Statistical Analyses

- Feature importance and non-linear effects were examined using CatBoost.
- Gender-stratified models were estimated to assess heterogeneous effects.
- Experimental data were analyzed using **ANOVA** under a **2 × 2 factorial design**.

---

## Repository Structure

```
├── data/
│   ├── raw/                # Original survey and experimental data (restricted access)
│   └── processed/          # Cleaned datasets used in analysis
├── scripts/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── model_catboost.py
│   └── statistical_tests.R
├── results/
│   ├── tables/
│   └── figures/
├── notebooks/
│   └── exploratory_analysis.ipynb
└── README.md
```

---

## Environment

- IDE: **PyCharm 2025.1.2**  
  Build: `#PY-251.26094.141`, built on June 10, 2025  
  Source revision: `0f4e379ff77b9`  
  Licensed to: fuzzes ally  
  You have a perpetual fallback license for this version.  
  Subscription is active until December 31, 2099.  
  Runtime version: `21.0.7+9-b895.130 aarch64 (JCEF 122.1.9)`  
  VM: OpenJDK 64-Bit Server VM by JetBrains s.r.o.  
  Toolkit: `sun.lwawt.macosx.LWCToolkit`  
  OS: macOS 14.6  
  GC: G1 Young Generation, G1 Concurrent GC, G1 Old Generation  
  Memory: 2048M  
  CPU Cores: 8  
  Metal Rendering: ON  
  Registry: `ide.experimental.ui=true`

## Reproducing the Analysis

### Requirements

- Python
  
# Machine Learning & Visualization Packages ⚡
This project relies on the following Python packages for machine learning, model training, and visualization.  
| Package    | Version |
|------------|---------|
| Boruta     | **⚡0.4.3⚡** |
| catboost   | **⚡1.2.8⚡** |
| lightgbm   | **⚡4.6.0⚡** |
| matplotlib | **⚡3.9.4⚡** |
| matplotlib-inline | **⚡0.1.7⚡** |
| numpy      | **⚡2.0.2⚡** |
| openpyxl   | **⚡3.1.5⚡** |
| optuna     | **⚡4.4.0⚡** |
| pandas     | **⚡2.3.3⚡** |
| plotly     | **⚡6.2.0⚡** |
| scikit-learn | **⚡1.6.1⚡** |
| scipy      | **⚡1.13.1⚡** |
| seaborn    | **⚡0.13.2⚡** |
| shap       | **⚡0.49.1⚡** |
| statsmodels | **⚡0.14.4⚡** |
| torch      | **⚡2.7.1⚡** |
| umap-learn | **⚡0.5.9.post2⚡** |
| xgboost    | **⚡2.1.4⚡** |
| PyALE      | **⚡1.2.0⚡** |
| pygam      | **⚡0.9.1⚡** |
| torchviz   | **⚡0.0.3⚡** |
| tqdm       | **⚡4.67.1⚡** |
> ⚠️ **Note:** Use these versions to ensure full reproducibility of the analysis in this project.

### Notes

This project was developed and tested in PyCharm 2025.1.2. It is recommended to use the same or a compatible environment to reliably reproduce the results.


### Run the main model

```bash
python scripts/model_catboost.py
```


## Ethical Considerations

This study uses anonymized secondary survey data (CGSS) and supplementary experimental data collected through informed consent. No personally identifiable information is included in this repository.

---
