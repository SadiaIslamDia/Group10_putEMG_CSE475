# Hand Gesture Recognition using Graph Neural Networks (putEMG)

This repository contains the complete implementation for recognizing hand gestures from surface Electromyography (sEMG) signals using Graph Neural Networks (GNN). The project follows a strict subject-independent evaluation protocol and explores model interpretability using Explainable AI (XAI) techniques (SHAP and LIME).

## Project Information
* **Group Name:** Group 10
* **Dataset:** [putEMG Dataset](https://biolab.put.poznan.pl/putemg-dataset/) (24 sEMG channels, 8 hand gesture classes)
* **Track:** Track 1 (Graph Neural Networks - GNN)
* **Core Focus:** Subject-Independent Generalization, GraphSAGE Optimization, and XAI (SHAP & LIME) without data leakage.

## Dataset Details
The dataset consists of 619,479 sliding windows extracted from 24 sEMG electrodes. We extracted 7 time-domain features (MAV, RMS, STD, VAR, WL, ZC, SSC) per channel, resulting in 168 input features per window. The dataset is highly imbalanced, with the 'Idle' gesture accounting for ~61% of the data. 

**Data Split (Strict Subject-Independent Protocol):**
* **Training:** 31 Subjects (435,797 windows)
* **Validation:** 6 Subjects (85,277 windows)
* **Held-out Test (Frozen):** 7 Subjects (98,405 windows)

## Requirements
To reproduce the experiments, install the following dependencies:
```bash
pip install torch torchvision
pip install torch-geometric
pip install pandas numpy scikit-learn matplotlib seaborn
pip install xgboost lightgbm
pip install shap lime joblib
