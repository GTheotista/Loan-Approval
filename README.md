# Loan Approval Classification Dataset: README

This repository contains the Loan Approval Classification dataset and an accompanying project pipeline for building machine learning models to predict loan approval status.

## Dataset Overview

**Source**: [Loan Approval Classification Data on Kaggle](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data/data)  
**Description**: This synthetic dataset is inspired by the original Credit Risk dataset, enriched with additional variables for analyzing Financial Risk in Loan Approval decisions. SMOTENC was applied to simulate new data points and enlarge the dataset. It includes both categorical and continuous features.

### Dataset Structure

| Column Name                     | Description                                                | Data Type    |
|---------------------------------|------------------------------------------------------------|--------------|
| `person_age`                    | Age of the person                                          | Float        |
| `person_gender`                 | Gender of the person                                       | Categorical  |
| `person_education`              | Highest education level                                    | Categorical  |
| `person_income`                 | Annual income                                              | Float        |
| `person_emp_exp`                | Years of employment experience                            | Integer      |
| `person_home_ownership`         | Home ownership status (e.g., rent, own, mortgage)         | Categorical  |
| `loan_amnt`                     | Loan amount requested                                      | Float        |
| `loan_intent`                   | Purpose of the loan                                        | Categorical  |
| `loan_int_rate`                 | Loan interest rate                                         | Float        |
| `loan_percent_income`           | Loan amount as a percentage of annual income              | Float        |
| `cb_person_cred_hist_length`    | Length of credit history in years                         | Float        |
| `credit_score`                  | Credit score of the person                                | Integer      |
| `previous_loan_defaults_on_file`| Indicator of previous loan defaults                       | Categorical  |
| `loan_status` (target variable) | Loan approval status: 1 = approved; 0 = rejected          | Integer      |

---

## Project Workflow

### Data Preparation
1. **Outlier Removal**: To handle extreme values and improve model performance.
2. **One-Hot Encoding**: Encoding categorical variables for compatibility with machine learning models.
3. **Standard Scaling**: Standardizing continuous features to improve model convergence.

---

### Model Development

**Models Tested**:
- Logistic Regression (`logreg`)
- K-Nearest Neighbors (`knn`)
- Decision Tree Classifier (`dt`)
- Random Forest Classifier (`rf`)
- XGBoost Classifier (`xgb`)
- LightGBM Classifier (`lgbm`)

**Model Evaluation Metric**:
- ROC AUC score was used to benchmark performance.

**Key Results**:
- **Tuned LightGBM ROC AUC**: **0.9763**

---

### Benchmarking and Model Improvement

1. **Oversampling**:
   - Applied SMOTENC to address class imbalance and improve minority class predictions.

2. **Hyperparameter Tuning**:
   - Conducted using grid search for optimal parameters.

3. **Performance Metrics (Tuned LightGBM)**:
   - **Class 0 (Negative Class)**:
     - Precision: **0.97**
     - Recall: **0.92**
     - F1-Score: **0.94**
   - **Class 1 (Positive Class)**:
     - Precision: **0.76**
     - Recall: **0.88**
     - F1-Score: **0.82**
   - **Overall Performance**:
     - Accuracy: **91%**
     - Macro Average (Precision, Recall, F1-Score): **0.87**, **0.90**, **0.88**
     - Weighted Average: Similar to macro metrics, reflecting class distribution.

---

### Conclusion

The **tuned LightGBM model** demonstrates exceptional performance in predicting loan approval status, especially for the majority class. While the recall for the minority class is high, precision is slightly lower, reflecting a trade-off often encountered in imbalanced datasets.

---

## Requirements
- Python 3.x
- Libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `matplotlib`, `seaborn`

---

## Future Improvements
1. Investigate other ensemble methods to further optimize minority class performance.
2. Explore feature engineering techniques to enhance predictive power.
3. Implement cost-sensitive learning for imbalanced datasets.

---
