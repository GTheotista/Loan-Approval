# Loan Approval Classification

**Tools:** Python
**Visualizations:** Matplotlib, Seaborn
**Dataset:**  [Loan Approval Classification Data](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data/data)

---

## I. Business Problem Understanding
Loan defaults pose a major risk to financial institutions, directly affecting profitability and overall portfolio stability. Early identification of high-risk applicants enables better credit decisions, minimizing potential losses and ensuring sustainable lending practices.

### Problem Statement
- Business Perspective: Loan defaults reduce revenue and increase risk exposure. A predictive model that classifies applicants into Default vs Non-Default is critical for effective credit risk management.
- Customer Perspective: Accurate assessment of creditworthiness ensures fair loan terms and prevents unnecessary rejection of low-risk applicants.

### Goals
- Develop a machine learning model to predict loan default based on applicant demographics, financial status, and credit history.
- Identify key features that influence the likelihood of default.
- Provide actionable insights to improve the loan approval and risk assessment process.

### Business Impact
- With a recall of ~0.91, the model can detect the majority of applicants likely to default.
- Enables proactive risk management and better credit policy design.
- Supports balanced loan portfolio growth while minimizing bad debt.

---

## II. Data Understanding & Preprocessing
### Dataset Source
Loan Approval Classification Data

### Key Steps
- Data Cleaning: Checked for missing values and handled anomalies.
- Feature Engineering:
     - Created age categories (Young, Adult, Mature, Senior).
     - Created income categories (Low, Medium, High, Very High).
     - Created employment experience categories (<1 year, 1–5 years, 5–10 years, 10–20 years, 20+ years).
- Handling Outliers:
     - Applied capping for extreme loan_amnt and credit_score values.
     - Used RobustScaler on skewed numerical features (loan amount, interest rate, loan-to-income ratio, credit history length).
- Encoding: Applied One-Hot Encoding to categorical variables.
- Pipeline: Combined preprocessing steps into a pipeline to avoid data leakage and ensure reproducibility.

---

## III. Modeling
### Algorithms Tested
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

### Best Model
LightGBM Classifier (Tuned)

### Performance (Test Set)
- Accuracy: ~0.90
- ROC-AUC: ~0.97
- Recall: ~0.91 (captures most defaults)
- F1-Score: ~0.80

### Key Predictors
- Loan Amount
- Credit Score
- Loan-to-Income Ratio
- Loan Interest Rate
- Credit History Length
- Home Ownership Status
- Previous Loan Defaults
- Loan Intent (Education, Venture, Home Improvement)
- Applicant’s Income Category

---

## IV. Conclusion and Recommendations
### Conclusion
- Tuned LightGBM provided the best trade-off between recall and ROC-AUC, making it effective in identifying high-risk applicants.
- Financial and credit-related variables (loan amount, credit score, debt-to-income ratio) are the most critical predictors of default.

### Recommendations
**Risk Management:**
- Deploy the model within the loan approval pipeline to flag high-risk applications.
- Adjust interest rates and credit terms based on applicant risk scores.

**Credit Policy:**
- Apply stricter evaluation to applicants with high loan-to-income ratios and low credit scores.
- Provide favorable terms to applicants with stable credit histories and home ownership.

**Customer Strategy:**
- Offer financial literacy and counseling to borderline applicants to reduce default risk.
- Design loan products tailored for different income and employment experience segments.

**Model Maintenance:**
- Retrain the model periodically to adapt to new borrower trends and economic changes.

---

## V. Next Steps
- Deployment: Build an API or dashboard to integrate the model into the loan decision-making system.
- Retraining: Continuously update and retrain the model as more data becomes available.
- Feature Expansion: Incorporate behavioral and transactional data for deeper credit risk profiling.
- Advanced Modeling: Extend the approach to predict probability of default (PD) and expected loss for more comprehensive risk management.
