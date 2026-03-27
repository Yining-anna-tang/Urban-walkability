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

## Reproducing the Analysis

### Requirements

- Python 3.9+
- R 4.2+
- catboost
- pandas
- numpy
- scikit-learn
- statsmodels


### Run the main model

```bash
python scripts/model_catboost.py
```


## Ethical Considerations

This study uses anonymized secondary survey data (CGSS) and supplementary experimental data collected through informed consent. No personally identifiable information is included in this repository.

---
