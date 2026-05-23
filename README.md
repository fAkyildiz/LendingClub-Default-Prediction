# LendingClub Loan Default Prediction

Binary classification project predicting whether a LendingClub loan will be **Fully Paid** or **Charged Off** (defaulted).

## Dataset

- **Source**: LendingClub accepted loans 2007–2018 Q4
- **Size**: 1.68 GB, 151 columns, ~1.35M usable rows (Fully Paid / Charged Off only)
- **Class imbalance**: ~80% Fully Paid / ~20% Charged Off
- Data files are excluded from this repo due to size — download from [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

## Project Structure

| File | Description |
|------|-------------|
| `01_eda.ipynb` | Exploratory Data Analysis |
| `02_preprocessing.ipynb` | Feature engineering & preprocessing |
| `03_modeling.ipynb` | Model training, evaluation & explainability |
| `LendingClub_Project_Proposal.pdf` | Original project proposal |
| `Presentation_Script.pdf` | Final presentation script |

## Workflow

### 1. EDA (`01_eda.ipynb`)
- Target variable analysis (class distribution, default rates by grade, purpose, term, state)
- Numerical feature distributions: FICO, interest rate, DTI, loan amount, annual income
- Missing value analysis (73 columns have missing data; 57 have >50% missing)
- Identification of **37 post-origination columns** to exclude (data leakage prevention)
- Comparison of rejected vs. accepted applications
- Engineered feature previews: `loan_to_income`, `credit_age_months`

### 2. Preprocessing (`02_preprocessing.ipynb`)
- Drop post-origination and ID columns
- Filter to Fully Paid / Charged Off loans
- Encode categorical variables
- Impute missing values
- Export processed dataset (1,345,310 rows × 133 columns)

### 3. Modeling (`03_modeling.ipynb`)
- Stratified 80/20 train/test split
- Three models trained with class imbalance handling:
  - Logistic Regression (`class_weight='balanced'`)
  - Random Forest (`class_weight='balanced_subsample'`)
  - XGBoost (`scale_pos_weight`)
- SMOTE oversampling comparison
- Threshold optimization for F1 maximization
- SHAP explainability
- Model vs. LendingClub grade-only baseline comparison

## Results

| Model | ROC-AUC | PR-AUC | F1 (t=0.5) |
|-------|---------|--------|------------|
| Logistic Regression | 0.7186 | 0.3828 | 0.4359 |
| Random Forest | 0.7175 | 0.3859 | 0.4326 |
| XGBoost | **0.7342** | **0.4095** | **0.4461** |
| XGBoost (optimal threshold) | 0.7342 | 0.4095 | 0.4485 |

**XGBoost** is the best model with a **+0.054 ROC-AUC lift** over LendingClub's own grade-only signal (0.7342 vs 0.6801).

## Top Features (SHAP)

1. `sub_grade` — strongest individual predictor
2. `term` — 60-month loans default significantly more than 36-month
3. `grade` — monotonically increasing default rate A→G
4. `dti` — debt-to-income ratio
5. `loan_to_income` — engineered feature
6. `acc_open_past_24mths` — recent credit activity
7. `int_rate` — higher rates strongly associated with default
8. `fico_range_low` — lower FICO scores predict default

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
imbalanced-learn
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap imbalanced-learn
```
