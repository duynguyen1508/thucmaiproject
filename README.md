# NTC Consumer Credit Risk Analysis

## 📌 Project Goals
This project explores **consumer credit risk modeling** in the **New-To-Credit (NTC)** segment.  
Our objectives are:  
- Perform **Exploratory Data Analysis (EDA)** to uncover patterns in borrower profiles.  
- Clean and engineer features (FICO bands, DTI ratios, utilization bands, etc.).  
- Build **machine learning models** to predict the likelihood of charge-off (`Ever_ChargeOff`).  
- Compare **Random Forest** and **XGBoost** performance with a focus on **precision** (minimizing false rejections of potential customers).  

---

## 🛠️ Technologies Used
- **Python 3.10+**  
- **Pandas / NumPy** → Data wrangling and feature engineering  
- **Matplotlib / Seaborn** → Visualization and EDA  
- **Scikit-Learn** → Data splitting, Random Forest, evaluation metrics  
- **XGBoost** → Gradient boosting model for imbalanced data  
- **Google Colab** → Development environment  

---

## 📊 Dataset Overview
The dataset contains consumer credit profiles, including:  
- **Demographics**: Age, Bankruptcy history, Collections  
- **Credit features**: FICO score, Revolving utilization, Student loan amounts, Unsecured debt  
- **Assets & Income**: Checking account balance, Reported income  
- **Target variables**:  
  - `Ever_ChargeOff` (binary)  
  - `ChargeOff_Balance` (continuous)  

**Key Stats after cleaning:**  
- Rows: ~N (after removing outliers with IQR)  
- Columns: 26+ engineered features  
- Missing values handled (e.g., imputing `FICO_V` with sentinel value 999)  

---

## 🔎 Exploratory Data Analysis (EDA)
1. **Correlation Analysis**:  
   - Strongest correlations were observed between `Ever_ChargeOff` and variables like **Revolving Utilization** and **Student Loan Amount**.  
   - Heatmaps highlighted both binary and continuous relationships with the target.  

2. **Outlier Detection**:  
   - Applied **IQR method** across continuous variables.  
   - Outliers removed to reduce skewness and improve model stability.  

3. **Distribution Plots**:  
   - FICO scores and income were right-skewed.  
   - Utilization ratios showed extreme values >100%.  

---

## 🧹 Data Cleaning & Feature Engineering
- **Missing values**:  
  - `FICO_V` imputed with `999`.  
- **Feature engineering**:  
  - `DTI` (Debt-to-Income ratio).  
  - **Banded variables**:  
    - FICO bands (Poor → Exceptional).  
    - Utilization bands (Very Low → Extreme).  
    - DTI bands (<36% → >50%).  
    - Loan amount bands (No debt → Very High).  
    - Asset bands (<500 → >2500).  
  - Added helper column `Accounts = 1` for count-based aggregation.  

---

## 🤖 Modeling Approach
Two models were trained:  

1. **Random Forest Classifier**  
   - `n_estimators = 100`  
   - `max_depth = 6`  
   - `class_weight = balanced`  

2. **XGBoost Classifier**  
   - `n_estimators = 100`  
   - `max_depth = 6`  
   - `scale_pos_weight` to correct for imbalance  

**Evaluation Strategy**:  
- 70/15/15 split into **Train / Validation / Test**.  
- Performance measured using **Precision, Recall, F1, and AUC**.  
- Decision threshold set at **0.7** to favor **Precision over Recall**.  

---

## 📈 Feature Importance
- **Random Forest**:  
  - Top drivers: Revolving Utilization, Student Loan Amount, Income, Checking Assets.  
- **XGBoost**:  
  - Similar importance ranking, but gave higher weight to interaction terms like DTI and Loan Amount.  

Both models confirm that **high utilization** and **low FICO** are primary predictors of charge-off.  

---

## 📊 Results Comparison

| Metric (Test Set)         | Random Forest | XGBoost |
|----------------------------|---------------|----------|
| **Precision (Charge-off)** | ~53%          | ~59%     |
| **Recall**                 | Lower         | Lower    |
| **AUC**                    | ~0.53–0.55    | ~0.58–0.60 |
| **Business Interpretation** | Balanced model, interpretable | More precise, but higher risk of overfitting |

**Takeaway**:  
- **Random Forest** provides a balanced baseline with interpretability.  
- **XGBoost** yields **higher precision**, which is strategically aligned with retaining more NTC customers (reducing false negatives at the cost of recall).  

---

## 📂 Output
Final cleaned dataset with engineered features exported to:  
```
/content/sample_data/NTC_Consumer_Credit_Data.csv
```

---

✅ Next steps could include:  
- Hyperparameter tuning (grid/random search).  
- Ensemble stacking with logistic regression.  
- Incorporating alternative data sources (e.g., telco, rent payments).  
