# Hand Gesture Recognition using Graph Neural Networks (putEMG)

This repository contains the complete implementation for recognizing hand gestures from surface Electromyography (sEMG) signals using Graph Neural Networks (GNN). The project follows a strict subject-independent evaluation protocol and explores model interpretability using Explainable AI (XAI) techniques, including SHAP and LIME.

## Project Information

- **Group Name:** Group 10
- **Dataset:** [putEMG Dataset](https://biolab.put.poznan.pl/putemg-dataset/) (24 sEMG channels, 8 hand gesture classes)
- **Track:** Track 1 (Graph Neural Networks - GNN)
- **Core Focus:** Subject-Independent Generalization, GraphSAGE Optimization, and XAI (SHAP & LIME) without data leakage.

## Dataset Details

The dataset consists of **619,479 sliding windows** extracted from 24 sEMG electrodes. We extracted 7 time-domain features (MAV, RMS, STD, VAR, WL, ZC, SSC) per channel, resulting in **168 input features per window**.

The dataset is highly imbalanced, with the **Idle** gesture accounting for approximately **61%** of the data.

### Data Split (Strict Subject-Independent Protocol)

- **Training:** 31 Subjects (435,797 windows)
- **Validation:** 6 Subjects (85,277 windows)
- **Held-out Test (Frozen):** 7 Subjects (98,405 windows)

## Requirements

To reproduce the experiments, install the following dependencies:

```bash
pip install torch torchvision
pip install torch-geometric
pip install pandas numpy scikit-learn matplotlib seaborn
pip install xgboost lightgbm
pip install shap lime joblib
```


## How to Run

The project is divided into three sequential tasks. Run the notebooks in the following order.

### 1. `Task1_EDA_and_Preprocessing.ipynb`

This notebook performs the initial data analysis and preprocessing.

- Performs Exploratory Data Analysis (EDA).
- Extracts the 7 relevant time-domain features:
  - MAV
  - RMS
  - STD
  - VAR
  - WL
  - ZC
  - SSC
- Drops redundant features such as IEMG.
- Prepares the dataset for Subject-Independent splitting.

### 2. `Task2_Baselines_and_Proposed_GNN.ipynb`

This notebook trains the traditional machine learning baselines and the initial proposed GNN model.

- Fits `VarianceThreshold` and `RobustScaler` strictly on the training subjects.
- Trains the following traditional ML baselines:
  - Logistic Regression
  - Decision Tree
  - k-NN
  - LightGBM
  - XGBoost
  - Random Forest
  - SVM
  - MLP
- Implements the initial proposed `Corr-GCN` model.
- Builds a Top-K=4 correlation graph dynamically from the training data.

### 3. Task 3 — Model Improvement, Ablation and Evaluation

Task 3 consists of three sequential Jupyter Notebooks. Run them in the following order.

#### 3.1 `Group10_putEMG_task3_improvement_ablation1.ipynb`

This notebook begins the improvement and ablation study of the proposed GNN model.

- Performs controlled experiments to improve the performance of the initial GNN model.
- Investigates the effect of different **Top-K neighbourhood sizes** (`K = 2, 4, 6, 8`).
- Compares **sparse Top-K graph construction** with a fully connected (dense) graph.
- Evaluates different **edge-weight modes**, including correlation-based and binary/unweighted edges.
- Identifies the most suitable graph construction configuration based on validation performance.
- Performs the initial stages of the controlled architecture and graph ablation study.

#### 3.2 `Group10_putEMG_task3_improvement_ablation2.ipynb`

This notebook continues the controlled improvement and ablation study using the best configuration obtained from the previous stage.

- Compares different GNN variants, including **GCN, GAT, and GraphSAGE**.
- Investigates different **GNN layer configurations and hidden sizes**.
- Tests regularization and optimization settings such as:
  - Dropout
  - Batch Normalization
  - Optimizer settings
  - Learning-rate scheduling
- Investigates different approaches for handling the highly imbalanced dataset.
- Evaluates **class-weighted and unweighted loss functions**.
- Tests different **early-stopping patience** settings.
- Selects the final GNN configuration based on the controlled ablation results.
- Prepares the selected configuration for the final cross-validation and held-out test evaluation.

#### 3.3 `Group10_putEMG_task3_improvement_ablation3.ipynb`

This notebook performs the final evaluation of the selected GNN configuration.

- Runs the finalized GNN configuration using the results obtained from the previous ablation stages.
- Performs **Subject-Grouped 5-Fold Cross Validation**.
- Compares the Final GNN with the best baseline model (MLP).
- Performs the **paired Wilcoxon signed-rank test** to evaluate whether the performance difference between the two models is statistically significant.
- Evaluates the finalized model on the **frozen held-out test subjects**.
- Reports the final classification performance using Accuracy, Balanced Accuracy, Macro Precision, Macro Recall, and Macro-F1.
- Generates the final evaluation results and confusion matrix for the held-out test set.

## Results & Findings

### 1. 5-Fold Cross Validation (Subject-Grouped)

The optimized Final GNN using GraphSAGE, Top-K=2, Unweighted edges, and 3 hidden layers was evaluated against the Best Baseline (MLP) over 5 strict subject-grouped folds.

| **Model** | **Macro-F1 Mean** | **Macro-F1 Std** | **Accuracy Mean** | **Balanced Acc. Mean** |
|---|---:|---:|---:|---:|
| **MLP (PyTorch)** | 0.5340 | 0.0422 | 0.6558 | 0.5815 |
| **Final GNN (GraphSAGE)** | 0.5143 | 0.0325 | 0.7243 | 0.5047 |

A paired Wilcoxon signed-rank test confirmed no statistically significant difference (**p = 0.1875**) between the Final GNN and the MLP baseline across the five subject-grouped folds.

### 2. Held-Out Test Performance

The Final GNN was evaluated strictly once on the **7 frozen test subjects** after finalizing the hyperparameters.

| **Model** | **Accuracy** | **Balanced Accuracy** | **Macro Precision** | **Macro Recall** | **Macro-F1** |
|---|---:|---:|---:|---:|---:|
| **Final GNN** | 0.7354 | 0.5075 | 0.5587 | 0.5075 | **0.5238** |

### 3. Explainability (XAI)

To ensure unbiased interpretation and avoid cherry-picking, one correct prediction and one wrong prediction were selected based on confidence scores closest to the median.

SHAP and LIME reference backgrounds were isolated to training subjects only to prevent data leakage.

- **Dominant Feature:** Both SHAP and LIME identified **Waveform Length (WL)** as the most critical metric for the model's decision-making process, followed by Standard Deviation (STD).
- **Correct Prediction (Pinch-Ring):** The model successfully identified the gesture using signals primarily from Electrodes 22, 10, and 13.
- **Incorrect Prediction (Flexion predicted as Fist):** The model showed confusion with **57% confidence** due to overlapping muscle activation patterns in the flexor muscles, heavily influenced by signals from Electrodes 14 and 6.

## Repository Structure

The repository follows the required Group-Dataset-Course structure.

```text
Group10_putEMG_CSE475/
│
├── README.md
│
├── report/
│   ├── task1/
│   │   └── Group10_putEMG_task1_report.pdf
│   │
│   ├── task2/
│   │   └── Group10_putEMG_task2_report.pdf
│   │
│   └── task3/
│       └── Group10_putEMG_task3_report.pdf
│
├── code/
│   ├── task1/
│   │   └── Task1_EDA_and_Preprocessing.ipynb
│   │
│   ├── task2/
│   │   └── Task2_Baselines_and_Proposed_GNN.ipynb
│   │
│   └── task3/
│       ├── Group10_putEMG_task3_improvement_ablation1.ipynb
│       ├── Group10_putEMG_task3_improvement_ablation2.ipynb
│       └── Group10_putEMG_task3_improvement_ablation3.ipynb
│
├── related_work/
│   ├── Group10_putEMG_related_work_table.pdf
│   │
│   └── papers/
│       ├── paper1.pdf
│       ├── paper2.pdf
│       ├── paper3.pdf
│       ├── paper4.pdf
│       └── paper5.pdf
│
└── models/
    ├── Group10_putEMG_best.pth
    ├── Group10_putEMG_preprocessing.joblib
    └── label_map.json
```

## Model and Preprocessing Files

The trained model and preprocessing files are stored in the `models/` directory.

- `Group10_putEMG_best.pth` - Final trained GNN model checkpoint.
- `Group10_putEMG_preprocessing.joblib` - Saved preprocessing pipeline.
- `label_map.json` - Mapping between gesture labels and class indices.

## Key Experimental Findings

The experiments were conducted using a strict subject-independent evaluation protocol, with training, validation, and held-out test subjects kept completely separate.

The Task 3 improvement and ablation study evaluated the proposed GNN through a series of controlled experiments covering graph construction, neighbourhood size, edge-weight configuration, GNN variants, architecture, regularization, optimization, class-imbalance handling, and early stopping.

The final GNN configuration used a **Mutual Information-based graph with Top-K=6 binary edges**, a **GraphSAGE variant with hidden layers [64, 128, 128]**, **Min+Max pooling**, and **Dropout=0.3**.

For final model comparison, Subject-Grouped **5-Fold Cross Validation** was performed against the best baseline model, MLP. A paired Wilcoxon signed-rank test produced a **p-value of 1.000**, indicating that the difference between the Final GNN and MLP was not statistically significant at the 0.05 significance level.

On the frozen held-out test set containing **7 unseen subjects and 98,405 windows**, the Final GNN achieved an **Accuracy of 75.46%** and a **Macro-F1 score of 0.5600**.

The class-wise analysis showed stronger performance on classes such as **Idle, Extension, and Flexion**, while the four pinch gestures, namely **Pinch-Index, Pinch-Middle, Pinch-Ring, and Pinch-Small**, remained more challenging due to overlapping muscle activation patterns.

## Explainable AI Findings

The XAI analysis was performed on the finalized GNN model to understand the model's decision-making process.

SHAP and LIME were used to analyze feature contributions while maintaining the leakage-safe evaluation protocol. The XAI reference/background data were restricted to training subjects to prevent information from the held-out test subjects from influencing the explanations.

The analysis focused on representative correct and incorrect predictions selected using a median-confidence based selection strategy to avoid cherry-picking.

The XAI analysis identified **Waveform Length (WL)** and **Standard Deviation (STD)** among the most influential sEMG features. Electrode-level analysis was also used to identify which electrodes contributed most strongly to the model's predictions.

The explanations provided additional insight into why the model performed better on some gesture classes while showing confusion among visually and physiologically similar pinch gestures.

## Reproducibility

All experiments follow a strict subject-independent evaluation protocol.

- Training subjects are used for model training and fitting preprocessing components.
- Validation subjects are used for model selection and hyperparameter tuning.
- The held-out test subjects remain completely frozen during model development.
- The final model is evaluated on the held-out test subjects only after the model configuration has been finalized.
- Graph construction and preprocessing information are derived only from the training data.
- SHAP and LIME reference/background data are restricted to training subjects to prevent data leakage.
- Subject-Grouped 5-Fold Cross Validation is used for robust evaluation of the final GNN and baseline model.

## Technologies Used

- Python
- PyTorch
- PyTorch Geometric
- Scikit-learn
- LightGBM
- XGBoost
- SHAP
- LIME
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook
## Dataset

The project uses the **putEMG Dataset**, containing surface Electromyography (sEMG) recordings for hand gesture recognition.

Dataset source: [putEMG Dataset](https://biolab.put.poznan.pl/putemg-dataset/)

## Group

**Group 10**

**Course:** CSE475

**Project Track:** Track 1 - Graph Neural Networks (GNN)
