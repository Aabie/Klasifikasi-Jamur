# Mushroom Classification

This project aims to classify mushrooms into two main classes: **edible** and **poisonous** using secondary data and various machine learning techniques.

## 1. Dataset Description

The dataset used is `secondary_data.csv`. This file contains morphological features of mushrooms such as shape, color, surface, and other characteristics relevant for classification.

## 2. Analysis Workflow

### a. Load Libraries and Data

- Using `pandas`, `seaborn`, and `matplotlib` for data exploration.
- Data is read from a CSV file with a `;` delimiter.

### b. Feature Engineering

- Mapping category codes to more descriptive labels (e.g., 'p' → 'poisonous', 'e' → 'edible').
- Replacing labels in several categorical columns like `cap-shape`, `cap-surface`, `cap-color`, etc.
- Handling missing values and removing features with too many missing values or those that are less informative.

### c. Exploratory Data Analysis (EDA)

- Visualizing the distribution of numerical and categorical features based on class using KDE plots and bar plots.
- Performing bivariate and multivariate analysis, including correlation between features after encoding.

### d. Data Preprocessing

- Removing duplicate data.
- Filling missing values in several columns with default values.
- Applying label encoding to categorical features.
- Splitting data into training and testing sets (80:20).

### e. Modeling and Evaluation

#### 1. Random Forest

- 5-fold cross-validation with SMOTE to handle data imbalance.
- Evaluating using metrics: accuracy, precision, recall, F1, and MCC.
- Saving the model for each fold.
- Evaluating on the test set and visualizing the confusion matrix.
- Analyzing feature importance.

#### 2. Gradient Boosting Machine (LGBM)

- Similar process to Random Forest, using LightGBM.
- Evaluating and visualizing metrics and feature importance.

#### 3. Extreme Gradient Boosting (XGBoost)

- Similar process to the two previous models.
- Evaluating and visualizing metrics and feature importance.

## 3. Folder Structure

```
Mushroom Classification/
│
├── secondary data/
│   ├── main.ipynb
│   └── secondary_data.csv
│
├── model_result/
│   ├── Random_Forest/
│   ├── LGBM/
│   └── XGBoost/
│
├── README.md
└── requirements.txt
```

## 4. How to Run

To run the analysis:

1. Open the `secondary data/main.ipynb` file.
2. Run all cells in the notebook.

## 5. Results and Evaluation

- Models are evaluated using cross-validation and on the test dataset.
- The main metrics used for evaluation are:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - MCC (Matthews Correlation Coefficient)
- Confusion matrix visualizations and feature importance are provided to help interpret the models.

### Example Cross-Validation Metrics (Random Forest)

**Train Metrics per Fold:**

|        | accuracy | precision | recall   | f1       | mcc      |
| ------ | -------- | --------- | -------- | -------- | -------- |
| Fold 1 | 0.998184 | 0.999166  | 0.997428 | 0.998296 | 0.996353 |
| Fold 2 | 0.998224 | 0.998864  | 0.997806 | 0.998335 | 0.996433 |
| Fold 3 | 0.998749 | 0.998941  | 0.998714 | 0.998827 | 0.997486 |
| Fold 4 | 0.998668 | 0.999092  | 0.998411 | 0.998751 | 0.997324 |
| Fold 5 | 0.998547 | 0.999091  | 0.998184 | 0.998638 | 0.997081 |

**Validation Metrics per Fold:**

|        | accuracy | precision | recall   | f1       | mcc      |
| ------ | -------- | --------- | -------- | -------- | -------- |
| Fold 1 | 0.998224 | 0.999091  | 0.997579 | 0.998335 | 0.996434 |
| Fold 2 | 0.997578 | 0.998183  | 0.997277 | 0.997730 | 0.995135 |
| Fold 3 | 0.998547 | 0.998789  | 0.998487 | 0.998638 | 0.997081 |
| Fold 4 | 0.998224 | 0.997883  | 0.998789 | 0.998336 | 0.996432 |
| Fold 5 | 0.996933 | 0.997878  | 0.996368 | 0.997123 | 0.993839 |

### Test

**Final Test Results for All Models (Random Forest):**

| Model  | Accuracy | Precision | Recall   | F1-Score | MCC      |
| ------ | -------- | --------- | -------- | -------- | -------- |
| Fold_1 | 0.997675 | 0.998565  | 0.997134 | 0.997849 | 0.995321 |
| Fold_2 | 0.998063 | 0.998804  | 0.997612 | 0.998208 | 0.996101 |
| Fold_3 | 0.998321 | 0.998091  | 0.998806 | 0.998448 | 0.996620 |
| Fold_4 | 0.998579 | 0.998567  | 0.998806 | 0.998687 | 0.997140 |
| Fold_5 | 0.998321 | 0.998329  | 0.998567 | 0.998448 | 0.996620 |

## 6. Additional Notes

- SMOTE (Synthetic Minority Over-sampling Technique) is used to address data imbalance in the target class.
- Features with a high percentage of missing values are removed from the analysis.
- Models and results from each cross-validation fold are stored in the `model_result/` folder according to their algorithm name.

## 7. Author

This project was carried out as part of an exploration of machine learning on mushroom classification data.
