  # 📊 NTC Consumer Credit Risk Analysis

This project analyzes **consumer credit risk** for the **New-to-Credit (NTC)** segment. It combines **Exploratory Data Analysis (EDA)**, **feature engineering**, and **machine learning** to predict the likelihood of customer charge-off, supporting business expansion strategies for lenders.  

---

## 🎯 Project Goals
- Understand credit risk patterns in **NTC customers**.  
- Explore and clean real-world credit and loan data.  
- Engineer new features such as **Debt-to-Income (DTI)** and utilization bands.  
- Build predictive models (**Random Forest, XGBoost**) for charge-off classification.  
- Balance **precision vs recall** to minimize false rejections while maximizing customer retention.  

---

## 🛠 Technologies Used
- **Python 3.8+**
- **pandas, numpy** – data analysis and manipulation  
- **matplotlib, seaborn** – visualizations  
- **scikit-learn** – machine learning models & evaluation  
- **xgboost** – gradient boosting algorithm  
- **Google Colab** – execution environment  

---

## 📂 Dataset Overview
The dataset (`v_credit_data_NTC_v51825.csv`) contains anonymized consumer credit profiles, including:  

- **FICO_V** – Credit score  
- **Income** – Annual income  
- **Unsecured_Debt** – Total unsecured debt  
- **Revolving_Util** – Credit utilization ratio  
- **Ever_ChargeOff** – Target variable (binary: ever charged-off)  
- **ChargeOff_Balance** – Exposure amount if default occurs  

⚠️ *The dataset is confidential and excluded from this repository.*  

---

## 🔍 Exploratory Data Analysis (EDA)
- Summary statistics and distributions  
- Correlation heatmaps with **Ever_ChargeOff** and **ChargeOff_Balance**  
- Histograms & KDE plots to visualize feature distributions  
- Binary variable frequency analysis (e.g., bankruptcy, collections)  

---

## 🧹 Data Cleaning & Feature Engineering
- **Outlier detection** with the IQR method  
- **Missing values**: imputed missing FICO scores with a placeholder (999)  
- **Feature Engineering**:
  - Debt-to-Income (DTI) ratio  
  - FICO bands (Poor → Exceptional)  
  - Utilization bands (Very Low → Extreme)  
  - DTI bands (Target → Very High)  
  - Loan amount and asset bands  
- Created additional categorical and banded features for interpretability  

---

## 🤖 Modeling Approach
- **Data Split**:  
  - 70% Training  
  - 15% Validation  
  - 15% Test  

- **Random Forest Classifier**  
  - Limited depth (`max_depth=6`)  
  - Balanced class weights for imbalanced targets  
  - Precision-focused thresholding (≥0.7 probability)  

- **XGBoost Classifier**  
  - Gradient-boosted trees with imbalance adjustment (`scale_pos_weight`)  
  - Improved precision compared to Random Forest  

- **Evaluation Metrics**  
  - Confusion Matrix  
  - Precision, Recall, F1-score  
  - ROC-AUC with curve visualization  

---

## 🌲 Feature Importance
Both **Random Forest** and **XGBoost** highlighted:  
- **FICO score**  
- **Revolving utilization**  
- **Debt-to-Income (DTI)** ratio  
- **Income** and **Loan Amount**  

Feature importance plots were generated to visualize the top 15 predictors.  

---

## 💡 Business Insight
- **~14.9% charge-off rate** in NTC sample (consistent with expectations).  
- **Random Forest** achieved ~53% precision on test data.  
- **XGBoost** improved precision to ~58.8%, with slightly reduced recall.  
- For lenders, prioritizing **precision** means fewer good customers are wrongly rejected, ensuring **higher retention and growth** in the NTC segment.  

---

## 🔮 Next Steps
- Deploy model as an API for **real-time credit scoring**.  
- Integrate **SHAP values** for model explainability.  
- Explore **deep learning** for capturing feature interactions.  
- Add **behavioral and transactional data** for stronger predictive power.  

---

## 📦 Installation
Clone the repo and install dependencies:
```bash
git clone https://github.com/<your-repo>/NTC_Consumer_Credit.git
cd NTC_Consumer_Credit
pip install -r requirements.txt
