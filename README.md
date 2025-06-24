# BitGrit Competition Solution

## Overview
This repository contains the complete solution for the BitGrit competition, implementing multiple machine learning models with hyperparameter optimization to achieve the best possible predictions.

## Project Structure
```
/bitgrit
├── /data
│   ├── train.csv           # Original training dataset
│   └── test.csv            # Original test dataset
├── /prepared               # Automatically created folder 
│   ├── train.csv           # Preprocessed training dataset
│   └── test.csv            # Preprocessed test dataset
├── know_data.ipynb         # Data preprocessing and feature engineering
├── models.ipynb            # Model training, optimization, and evaluation
├── requirements.txt        # Required Python libraries and versions
├── the_best.csv            # Final predictions on test set
├── README.md               # This file
└── model 77_75.txt         # LightGBM, XGBoost, Catboost parametrs from the best solution
```

## Environment Details

### System Configuration
- **Operating System**: Windows 11 Education Version 24H2 OS build: 26100.4351
- **RAM**: 32 GB Patriot 4133 MHz
- **Storage**: 1.75 GB free space
- **CPU**: Intel Core i7-9700K
- **GPU**: NVIDIA RTX 2080

### Python Environment
- **Python Version**: 3.12.3
- **Development Environment**: Jupyter Notebook
- **Package Manager**: pip

## Required Libraries
Install all required dependencies using:
```bash
pip install -r requirements.txt
```

The main libraries used include:
- pandas, numpy (data manipulation)
- scikit-learn (machine learning models and metrics)
- lightgbm, xgboost, catboost (gradient boosting models)
- matplotlib, seaborn (visualization)
- Additional libraries as specified in requirements.txt

## Data Files Used

### Input Data
- **train.csv**: Original training dataset containing features and target variable
- **test.csv**: Original test dataset for final predictions (without target variable)

### Processed Data
- **prepared/train.csv**: Preprocessed training dataset with engineered features
- **prepared/test.csv**: Preprocessed test dataset with engineered features

## Step-by-Step Reproduction Instructions

### 1. Environment Setup
```bash
# Clone or download the project
# Navigate to the project directory
cd bitgrit

# Install dependencies
pip install -r requirements.txt

# Start Jupyter Notebook
jupyter notebook
```

### 2. Data Preprocessing
```bash
# Open and run the data preprocessing notebook
# This will process the raw data and save it to /prepared folder
jupyter notebook know_data.ipynb
```

**What happens in know_data.ipynb:**
- Loads raw data from `/data/train.csv` and `/data/test.csv`
- Performs data cleaning and preprocessing
- Engineers new features
- Handles missing values
- Creates `/prepared` folder (if does not exist)
- Saves processed data to `/prepared/train.csv` and `/prepared/test.csv`

### 3. Model Training and Prediction
```bash
# Open and run the models notebook
jupyter notebook models.ipynb
```

**What happens in models.ipynb:**
- Loads preprocessed data from `/prepared` folder
- Implements multiple models: LightGBM, XGBoost, CatBoost
- Creates StackingClassifier with LogisticRegression as meta-learner
- Performs hyperparameter optimization for each model
- Evaluates models using ROC curves, confusion matrices, and learning curves
- Generates final predictions and saves to `stacking_lgb_xgb_cat.csv`
- Additional comparison with `the_best.csv`

### 4. Final Output
The best predictions were saved as `stacking_lgb_xgb_cat.csv` and manually moved to the root directory as `the_best.csv`.

## Data Processing Details

### Data Preprocessing Steps
The `know_data.ipynb` notebook performs the following preprocessing steps:
1. **Data Loading**: Reads raw CSV files from `/data` folder
2. **Data Exploration**: Analyzes data structure, missing values, and distributions
3. **Feature Engineering**:
   - Uses PCA for `job_desc` columns to reduce data dimensionality
   - Splits `job_posted_date` on column `month` and `year`
   - Creates `job_region` column based on the `job_state` and finally applies one-hot encoding on this new column
4. **Data Cleaning**: 
   - Handles missing values in `job_state`
5. **Data Validation**: Ensures data quality and consistency
   - Uses PSI to check if train and test dataset is similar in terms of value ranges, distribution
6. **Data Export**: Saves processed data to `/prepared` folder

## Algorithm Details

### Models Implemented
1. **LightGBM**: Gradient boosting framework optimized for speed and memory efficiency
2. **XGBoost**: Extreme gradient boosting with regularization
3. **CatBoost**: Gradient boosting with categorical feature handling
4. **StackingClassifier**: Meta-ensemble combining all base models with LogisticRegression

### Main Hyperparameters

#### CatBoost
- `iterations`: 137,
- `learning_rate`: 0.10095947308778701,
- `depth`: 7,
- `l2_leaf_reg`: 0.01124311885824868,
- `loss_function`: MultiClass,
- `random_seed`: 42,
- `logging_level`: Silent,
- `bagging_temperature`: 1.0,
- `early_stopping_rounds`: 20,
- `cat_features`: job_title, feature_1,
- `thread_count`: 1

#### XGBoost
- `objective`: multi:softprob,
- `base_score`: None,
- `booster`: None,
- `callbacks`: None,
- `colsample_bylevel`: None,
- `colsample_bynode`: None,
- `colsample_bytree`: None,
- `device`: None,
- `early_stopping_rounds`: None,
- `enable_categorical`: True,
- `eval_metric`: None,
- `feature_types`: None,
- `feature_weights`: None,
- `gamma`: 0.6551776163530296,
- `grow_policy`: None,
- `importance_type`: None,
- `interaction_constraints`: None,
- `learning_rate`: 0.07315523763489679,
- `max_bin`: 219,
- `max_cat_threshold`: None,
- `max_cat_to_onehot`: None,
- `max_delta_step`: None,
- `max_depth`: 10,
- `max_leaves`: 23,
- `min_child_weight`: None,
- `missing`: nan,
- `monotone_constraints`: None,
- `multi_strategy`: None,
- `n_estimators`: 118,
- `n_jobs`: 1,
- `num_parallel_tree`: None,
- `random_state`: 42,
- `reg_alpha`: 0.013705730913738874,
- `reg_lambda`: 0.012219537217204616,
- `sampling_method`: None,
- `scale_pos_weight`: None,
- `subsample`: 0.9988236950762315,
- `tree_method`: hist,
- `validate_parameters`: None,
- `verbosity`: None,
- `use_label_encoder`: None,
- `gpu_id`: None,
- `predictor`: None,
- `num_class`: 3

#### LightGBM
- `boosting_type`: gbdt,
- `class_weight`: None,
- `colsample_bytree`: 0.9333584928887282,
- `importance_type`: split,
- `learning_rate`: 0.23332896051177257,
- `max_depth`: 12,
- `min_child_samples`: 20,
- `min_child_weight`: 0.001,
- `min_split_gain`: 0.0,
- `n_estimators`: 142,
- `n_jobs`: 1,
- `num_leaves`: 34,
- `objective`: multiclass,
- `random_state`: 81,
- `reg_alpha`: 0.0,
- `reg_lambda`: 1,
- `subsample`: 0.9720591168744056,
- `subsample_for_bin`: 200000,
- `subsample_freq`: 0,
- `num_class`: 3,
- `boosting`: gbdt,
- `metric`: None,
- `verbose`: -1,
- `min_data_in_leaf`: 6
- `force_col_wise`: True,
- `deterministic`: True,
- `enable_bundle`: False

#### StackingClassifier
- **Base Models**: LightGBM, XGBoost, CatBoost
- **Meta-Learner**: LogisticRegression
- **Cross-Validation**: 6-fold stratified CV for base model training

### Hyperparameter Optimization
- **Method**: optuna with BasePruner and TPESampler
- **Cross-Validation**: 5-fold stratified cross-validation
- **Evaluation Metric**: accuracy

## Model Evaluation

### Evaluation Metrics
- **ROC Curves**: Area Under Curve (AUC) analysis
- **Confusion Matrices**: Classification accuracy assessment
- **Learning Curves**: Training and validation performance over time
- **Classification Report**: Model performance across most important metrics

## Execution Time
- **Data Preprocessing notebook**: Approximately 18 to 22 seconds
- **Model Training notebook (without hyperparameter optimization)**: Approximately 90 seconds

### Model Performance
The StackingClassifier achieved the best performance by combining the strengths of different gradient boosting algorithms. Each base model captures different aspects of the data patterns, while the meta-learner optimally combines their predictions.

### Feature Importance
Key features identified through model analysis include:
- all PCA components, especially `PCA_0`
- `feature_2`, which consistently ranked first in feature importance bar plots
- `feature_10`, where the analysis of NaN values during vacation months provided significant insight into the `Low` salary category

### Hyper-parameter Importance
Key Hyper-parameters identified through model analysis include:
- `C` in LogisticRegression
- `cv` number of stratified folds in StackingClassifier
- `max_depth`, `learning_rate`, `iterations` were crucial in boosting algorithms
- LogisticRegression coefficients and class separation are included in file `models.ipynb` for better understanding and interpretation

### Reproducibility Notes
- Random seeds are set for reproducible results in every model, train_test_split
- All preprocessing steps are documented and automated
- Model training follows deterministic procedures where possible
- To avoid thread scheduling variability across platforms `n_jobs`, or `thread_count` are set to 1

## Troubleshooting

### Common Issues
1. **Missing Dependencies**: Run `pip install -r requirements.txt`
2. **Data Path Issues**: Ensure proper folder structure as described above
3. **Jupyter Kernel Issues**: Restart kernel if notebooks hang
4. **LightGBM Issues**: Despite setting random_state, checking C++ compiler on my laptop (in the same configuration as on PC) LightGBM makes it impossible to replicate StackingClassifier performance from my PC (public score on PC -> 77.75).

### Support
For any issues with reproduction, please check:
1. All files are in correct locations
2. All dependencies are installed
3. Sufficient system resources are available
4. Python version compatibility (3.12.3)

---

**Note**: This solution was developed and tested on the specified environment. Results may vary slightly on different hardware configurations, but the overall approach and methodology remain consistent.