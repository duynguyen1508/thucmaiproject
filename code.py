
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import drive
drive.mount('/content/drive')
# Load the credit profile data
df = pd.read_csv('/content/v_credit_data_NTC_v51825 (1) (1).csv')
print("Data loaded successfully!")

# --- EDA Framework ---

# Set options to display all columns and a large number of rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print("\n--- EDA: Exploring Credit Risk Data ---")

# 1. Discuss the concept of consumer credit risk and define how it is calculated.
print("\n1. Credit Risk Concept and Calculation:")
print("Credit risk is the possibility of a borrower failing to repay a loan or meet their debt obligations. It's a crucial concern for lenders as it directly impacts their profitability.")
print("Credit risk assessment often involves evaluating a borrower's creditworthiness based on factors like credit history, repayment capacity, income, loan terms, and collateral.")
print("Quantifying credit risk can involve calculating metrics such as:")
print("- Probability of Default (PD%): The likelihood that a borrower will default on their debt.")
print("- Net Credit Loss ($NCL%): The total amount the lender is at risk for at the time of default.")

# 2. Explore the credit data
print("\n2. Exploring the Credit Data:")
print("\nFirst, let's get a general overview of the data:")
print(df.head())
print("\nDataframe information:")
df.info()
print("\nSummary statistics of numerical features:")
print(df.describe())
print("\nValue counts of categorical features:")
for column in df.select_dtypes(include='object').columns:
    print(f"\nValue counts for column '{column}':")
    print(df[column].value_counts())

# 3. Crosstab and pivot tables with two target variables
print("\n3. Correlation Heatmaps with Target Variables Ever_ChargeOff and ChargeOff_Balance:")
print("\nTo simplify the analysis of correlations, we will generate two heatmaps showing the correlation of all numerical variables with each of the target variables.")

# Assuming 'df' is your DataFrame and 'Ever_ChargeOff' and 'ChargeOff_Balance' are your target columns
target_binary = 'Ever_ChargeOff'
target_continuous = 'ChargeOff_Balance'

numerical_df = df.select_dtypes(include=np.number).copy()

# Correlation with Ever_ChargeOff
correlation_with_ever_chargeoff = numerical_df.corr()[target_binary].sort_values(ascending=False)

# Display the correlation table for Ever_ChargeOff
print("\nCorrelation Table with Ever_ChargeOff:")
print(correlation_with_ever_chargeoff.to_frame())

# Correlation heatmap for Ever_ChargeOff
plt.figure(figsize=(8, 10))
sns.heatmap(correlation_with_ever_chargeoff.to_frame(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title(f'Correlation with Ever_ChargeOff')
plt.show()

# Correlation with ChargeOff_Balance
correlation_with_chargeoff_balance = numerical_df.corr()[target_continuous].sort_values(ascending=False)

# Display the correlation table for ChargeOff_Balance
print("\nCorrelation Table with ChargeOff_Balance:")
print(correlation_with_chargeoff_balance.to_frame())

# Correlation heatmap for ChargeOff_Balance
plt.figure(figsize=(8, 10))
sns.heatmap(correlation_with_chargeoff_balance.to_frame(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title(f'Correlation with ChargeOff_Balance')
plt.show()

# 4. Outliers in credit data
print("\n4. Outliers in Credit Data:")
print("\nOutliers are data points that significantly deviate from the rest of the data. In credit data, outliers can represent unusual applicant profiles or errors in data entry. They can skew statistical analyses and affect machine learning model performance.")

# 5. Finding outliers with cross tables
print("\n5. Finding Outliers with Cross Tables:")
print("\nWhile cross tables primarily show relationships between categorical variables, unusual distributions within categories might hint at outliers in related numerical variables.")
print("For example, if a specific category has a very high or low average loan amount compared to others, it could indicate potential outliers.")

print("\nValue Frequency of Binary Columns:")
for column in df.columns:
    if df[column].nunique() == 2:
        print(f"\nValue counts for binary column '{column}':")
        print(df[column].value_counts())
print("\nSome noticeable insight: 1.45% went bankruptcy before, 20.45% ever collection, meanwhile, 14.15% ever charge-off, reasonable for NTC, young adult")

print("\nExamining Distributions with Histograms and KDE Plots:")
for column in numerical_df.columns:
  if numerical_df[column].nunique() > 2:
     plt.figure(figsize=(8, 6))
     sns.histplot(numerical_df[column], kde=True)
     plt.title(f'Distribution of {column}')
     plt.show()
  else:
      print(f"\nSkipping box plot for '{column}' as it is binary.")

print("\nFor column with null values like FICO, chargeoff_balance, the histogram only distibution for populated values")

print("\nHistogram is great to give overview for how values are distributed, but need 1.5Q Method to effetively identify outliner")
print("\n6.Identifying Outliers using IQR Method (Non-Binary Columns):")
for column in numerical_df.columns:
    if numerical_df[column].nunique() > 2:
        Q1 = numerical_df[column].quantile(0.25)
        Q3 = numerical_df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_iqr = numerical_df[column][(numerical_df[column] < lower_bound) | (numerical_df[column] > upper_bound)]
        if not outliers_iqr.empty:
            print(f"\nPotential outliers in '{column}' (IQR method):\n{outliers_iqr.head()}") # head function only shows 5 examples of the outliners
        else:
            print(f"\nNo significant outliers found in '{column}' using IQR method.")
    else:
        print(f"\nSkipping IQR outlier detection for binary column '{column}'.")

print("\n7. Removing Outliers (IQR Method, Non-Binary Columns):")
original_row_count = len(df)
rows_removed_per_column = {}
df_cleaned = df.copy()  # Create a copy of the DataFrame to avoid modifying the original during iteration

for column in numerical_df.columns:
    if numerical_df[column].nunique() > 2:
        Q1 = df_cleaned[column].quantile(0.25)
        Q3 = df_cleaned[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_mask = (df_cleaned[column] < lower_bound) | (df_cleaned[column] > upper_bound)
        outlier_indices = df_cleaned[outliers_mask].index
        rows_removed = len(outlier_indices)

        if rows_removed > 0:
            df_cleaned = df_cleaned.drop(outlier_indices)
            rows_removed_per_column[column] = rows_removed
            print(f"\nRemoved {rows_removed} rows with IQR outliers in column '{column}'.")
        else:
            print(f"\nNo IQR outliers found in column '{column}'.")
    else:
        print(f"\nSkipping outlier removal for binary column '{column}'.")

print(f"\n--- Summary of Rows Removed ---")
total_rows_removed = original_row_count - len(df_cleaned)
print(f"Original number of rows: {original_row_count}")
print(f"Number of rows after outlier removal: {len(df_cleaned)}")
print(f"Total number of rows removed: {total_rows_removed}")
print("\nRows removed per column (if any):")
for column, count in rows_removed_per_column.items():
    print(f"- '{column}': {count}")

print("\nCleaned DataFrame (first 5 rows):")
print(df_cleaned.head())

# 7. Risk with missing data in loan data
print("\n8. Risk with Missing Data in Loan Data:")
print("\nMissing data is a common problem in real-world datasets. In loan data, missing values can introduce bias and reduce the accuracy of analyses and models.")
print("The risk associated with missing data depends on the extent and pattern of missingness. For example:")
print("- Missing values in crucial features like income or credit score can significantly impact credit risk assessment.")
print("- If missingness is systematic (related to other variables), it can introduce bias.")
print("- High percentages of missing values in a column might render machine learning model prediction feature unreliable.")

# 8. Replacing missing credit data
print("\n9. Replacing Missing Credit Data:")
print("\nSeveral techniques can be used to handle missing data:")
print("- Imputation: Filling missing values with estimated values (e.g., mean, median, mode).")
print("- More sophisticated imputation techniques (e.g., using machine learning models).")

print("\nLet's check for missing values:")
print(df.isnull().sum().sort_values(ascending=False))

from sklearn.model_selection import train_test_split

# Assuming your DataFrame after cleaning is named 'df_cleaned'

print("\n10. Split data into Train, Validation, and Test Data:")

# Split the data into training, validation, and test sets
train_df, temp_df = train_test_split(df_cleaned, test_size=0.3, random_state=42) # 30% go to validation-test, 70% remain for training
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)     #break-down 50/50 of 30% for validation and test

print(f"Training data shape: {train_df.shape}")
print(f"Validation data shape: {val_df.shape}")
print(f"Test data shape: {test_df.shape}")

print("\n--- Explanation of Data Splitting ---")
print("""
We split our data into three distinct datasets: training, validation, and test. This is a fundamental practice in machine learning to build robust and reliable models.

**1. Training Data:**
   - This is the largest portion of our data and the one that the machine learning model directly learns from.
   - The model will adjust its internal parameters based on the patterns and relationships it finds in the training data.
   - Think of it as the 'textbook' the model studies to understand the underlying concepts.

**2. Validation Data:**
   - This dataset is used to tune the model's hyperparameters and to get an unbiased estimate of the model's performance *during* the training process.
   - Hyperparameters are settings of the model that are not learned from the data but are set prior to training (e.g., the learning rate of an algorithm, the depth of a decision tree).
   - By evaluating the model on the validation set after each training epoch or after trying different hyperparameter settings, we can see how well the model generalizes to unseen data and avoid overfitting (where the model learns the training data too well and performs poorly on new data).
   - The validation set acts as a 'practice exam' that helps us make adjustments to the model before the final evaluation.

**3. Test Data:**
   - This is a completely separate dataset that the model *never* sees during the training or hyperparameter tuning phases.
   - It serves as the final, unbiased evaluation of the model's performance on completely new, unseen data.
   - The test set simulates how well the model would perform in a real-world scenario.
   - We only evaluate the model on the test set *once*, after we have finalized our model through training and validation.
   - Think of the test set as the 'final exam' that gives us a true measure of the model's capabilities.

By using this three-way split, we can build a model that not only learns from the data but also generalizes well to new data and provides a reliable estimate of its real-world performance.
""")

# Impute missing FICO in the training data with a special value (e.g., 0)
special_fico_value = 999
train_df['FICO_Imputed'] = train_df['FICO_V'].fillna(special_fico_value).astype(int)

# Verify the imputation in the training data
print("\nMissing FICO values in training data before imputation:", train_df['FICO_V'].isnull().sum())
print("Missing FICO values in training data after imputation:", train_df['FICO_Imputed'].isnull().sum())
print("First few rows of training data with imputed FICO:")
print(train_df[['FICO_V', 'FICO_Imputed', 'No_Hit']].head())

# For demonstration, let's also do the same for validation and test sets *independently*
# In a real project, you would apply the *same* special value to these as well.
val_df['FICO_Imputed'] = val_df['FICO_V'].fillna(special_fico_value).astype(int)
test_df['FICO_Imputed'] = test_df['FICO_V'].fillna(special_fico_value).astype(int)

print("\nMissing FICO values in validation data after imputation:", val_df['FICO_Imputed'].isnull().sum())
print("Missing FICO values in test data after imputation:", test_df['FICO_Imputed'].isnull().sum())

#\nNow check Ever_ChargeOff rate in train data
print("\nValue Frequency of Ever_ChargeOff in Train Data:")
for column in train_df.columns:
    if column == 'Ever_ChargeOff':
        print(f"\nValue counts for Ever_ChargeOff '{column}':")
        print(train_df[column].value_counts())
print("\nSome noticeable insight: 14.93% ever charge-off, reasonable for NTC, young adult")
# remember to check whether undersampling for training test is needed
# At 4%, we're in the moderate imbalance zone. No need to undersample — as 20% is isually considered as stastically important — but we still need to address imbalance through model-aware techniques.

###### END of EDA #######

###### --- MODELING: Predicting Ever_ChargeOff --- ######

print("\n=== Step 11: Building ML Model to Predict Ever_ChargeOff ===")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

# Drop target and non-informative columns
# Also drop FICO_V and No_Hit to avoid redundancy (FICO_Imputed encodes both)
drop_cols = ['Ever_ChargeOff', 'FICO', 'No_Hit']
feature_cols = [col for col in train_df.columns if col not in drop_cols]

# 2. Prepare X and y
X_train = train_df[feature_cols]
y_train = train_df[target_binary]

X_val = val_df[feature_cols]
y_val = val_df[target_binary]

X_test = test_df[feature_cols]
y_test = test_df[target_binary]

# 3. Fit a Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100,    #  Train 100 decision trees in the ensemble. More trees improve stability and generalization. 100 is a common, reliable starting point for performance without being too slow.
    max_depth=6,         #  Each tree can go only 6 levels deep. Prevents overfitting by limiting tree complexity. Shallow trees generalize better. Helps model stay interpretable and faster. Depth 6 is a good balance for our credit data (likely enough to model interactions without memorizing).
    class_weight='balanced',  # Automatically adjusts weights to compensate for class imbalance. Without balancing, the model might ignore the minority class. This tells the algorithm to treat both classes as equally important by assigning higher weight to the rare class.
    random_state=42  # fixed seed for reproducibility
)
rf_model.fit(X_train, y_train)

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Train a shallow decision tree for easy visualization
tree = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
tree.fit(X_train, y_train)

# Plot the decision tree
plt.figure(figsize=(14, 8))
plot_tree(tree, feature_names=X_train.columns, class_names=["No Charge-off", "Charge-off"], filled=True)
plt.title("Sample Decision Tree (Max Depth = 3)")
plt.show()
#from the plot, at the root, we see gini at 0.5 meaning the highest uncertainty, getting down the tree, we see gini decrease

# 4. Predict the target variable "Ever_ChargeOff" on validation data
val_probs = rf_model.predict_proba(X_val)[:, 1]
val_pred_thresh = (val_probs >= 0.7).astype(int)
# for each application, model will print out the probability of default as a Percentage number:
# Use threshold 0.7 for classification. Anyone with a (default 50%) 70% or higher chance of charge-off is now flagged as a predicted charge-off (class 1)

# 5. Evaluate model performance using thresholded predictions
print("\n--- Validation Performance ---")
print(confusion_matrix(y_val, val_pred_thresh))
print(classification_report(y_val, val_pred_thresh, digits=3))

val_auc = roc_auc_score(y_val, val_probs)
print(f"Validation AUC: {val_auc:.3f}")

# 6. ROC Curve (unchanged, uses probabilities)
fpr, tpr, thresholds = roc_curve(y_val, val_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {val_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve on Validation Set")
plt.legend()
plt.grid()
plt.show()

# 7. Evaluate on Test Set (can remain unchanged if no thresholding applied here)
test_preds = rf_model.predict(X_test)
test_probs = rf_model.predict_proba(X_test)[:, 1]

print("\n--- Test Set Performance ---")
print(confusion_matrix(y_test, test_preds))
print(classification_report(y_test, test_preds, digits=3))
print(f"Test AUC: {roc_auc_score(y_test, test_probs):.3f}")
print("\nAs our intent is to support business expansion into the New-To-Credit (NTC) customer segment, our model with decent 53% Precision with test data, is now optimized to prioritize Precision.")
print("This strategy helps us retain more potential NTC customers, which is crucial for growth and market penetration, lthough this comes at the cost of lower Recall, it's an intentional trade-off to avoid missing out on valuable new customers.")

# 8. Feature importance (unchanged)
importances = rf_model.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp_df.head(15), x='Importance', y='Feature')
plt.title("Top 15 Feature Importances (Random Forest)")
plt.tight_layout()
plt.show()

# XGBoost: nureau network, cautious: overfiting, based off current dataset: 26 variables

# compare Random Forest and XGBoost (which has higher precision & AUC)
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

# Assuming X_train, y_train, X_val, y_val, X_test, y_test, and feature_cols are defined earlier

# 3. Fit an XGBoost Classifier
# Calculate scale_pos_weight for class imbalance (ratio of negative to positive samples)
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

xgb_model = XGBClassifier(
    n_estimators=100,    # Train 100 trees in the ensemble
    max_depth=6,         # Each tree can go only 6 levels deep to prevent overfitting
    scale_pos_weight=scale_pos_weight,  # Adjusts for class imbalance
    random_state=42,     # Fixed seed for reproducibility
    eval_metric='logloss'    # Standard metric for binary classification
)
xgb_model.fit(X_train, y_train)

# Train a shallow decision tree for visualization
tree = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
tree.fit(X_train, y_train)

# Plot the decision tree
plt.figure(figsize=(14, 8))
plot_tree(tree, feature_names=X_train.columns, class_names=["No Charge-off", "Charge-off"], filled=True)
plt.title("Sample Decision Tree (Max Depth = 3)")
plt.show()

# 4. Predict the target variable "Ever_ChargeOff" on validation data
val_probs = xgb_model.predict_proba(X_val)[:, 1]
val_pred_thresh = (val_probs >= 0.7).astype(int)  # Threshold of 0.7 for classification

# 5. Evaluate model performance using thresholded predictions
print("\n--- Validation Performance ---")
print("Confusion Matrix:")
print(confusion_matrix(y_val, val_pred_thresh))
print("\nClassification Report:")
print(classification_report(y_val, val_pred_thresh, digits=3))

val_auc = roc_auc_score(y_val, val_probs)
print(f"Validation AUC: {val_auc:.3f}")

# 6. ROC Curve
fpr, tpr, thresholds = roc_curve(y_val, val_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {val_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve on Validation Set")
plt.legend()
plt.grid()
plt.show()

# 7. Evaluate on Test Set
test_probs = xgb_model.predict_proba(X_test)[:, 1]
test_pred_thresh = (test_probs >= 0.7).astype(int)  # Apply same threshold as validation

print("\n--- Test Set Performance ---")
print("Confusion Matrix:")
print(confusion_matrix(y_test, test_pred_thresh))
print("\nClassification Report:")
print(classification_report(y_test, test_pred_thresh, digits=3))
print(f"Test AUC: {roc_auc_score(y_test, test_probs):.3f}")

print("\nAs a start-up aiming to expand and capture the NTC customer segment, our model achieves ~58.8% precision on test data, While this comes at the cost of lower Recall, it's an acceptable trade-off for minimizing lost opportunities in early growth stage.")
print("This reduces false rejections, we ensure that when the model flags someone as high-risk (charge-off), it's highly confident, helping us retain more potential customers during market expansion.")

# 8. Feature Importance
importances = xgb_model.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp_df.head(15), x='Importance', y='Feature')
plt.title("Top 15 Feature Importances (XGBoost)")
plt.tight_layout()
plt.show()

print(df_cleaned.columns)

# 1. Create DTI column: debt divided by income
# Assuming 'Debt' and 'Income' columns exist and are numeric
df_cleaned['DTI'] = df_cleaned['Unsecured_Debt'] / df_cleaned['Income']

# Handle possible division by zero or missing values if needed
#df_cleaned['DTI'] = df_cleaned['DTI'].replace([np.inf, -np.inf], np.nan)
#df_cleaned['DTI'] = df_cleaned['DTI'].fillna(0)  # or any imputation strategy

# 2. Create 'Accounts' column with value 1
df_cleaned['Accounts'] = 1

# 3. Create bands for  FICO, Utilization, DTI , Loan Amount, and Asset
# Define function to create bins and labels for convenience
def create_bins(series, bins, labels):
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True, right=False)

# Create bands
#FICO Band
fico_bins = [300, 580, 670, 740, 800, 850]
fico_labels = ['Poor', 'Fair', 'Good', 'Very Good', 'Exceptional']
df_cleaned['FICO'] = create_bins(df_cleaned['FICO_V'], fico_bins, fico_labels)

# Create FICO_Band_10pts: 10-point intervals
fico10_bins = list(range(301, 850, 10))  # 300, 310, ..., 850
fico10_labels = [f"{i}-{i+9}" for i in fico10_bins[:-1]]
df_cleaned['FICO_Band'] = create_bins(df_cleaned['FICO_V'], fico10_bins, fico10_labels)

#Utilization Band
util_bins = [0, 0.1, 0.3, 0.6, 1.0, np.inf]
util_labels = [
    'Very Low (<10%)',
    'Low (10-30%)',
    'Moderate (30-60%)',
    'High (60-100%)',
    'Extreme (>100%)'
]
df_cleaned['Utilization_Band'] = create_bins(df_cleaned['Revolving_Util'], util_bins, util_labels)

#DTI Band
dti_bins = [0, 0.36, 0.41, 0.43, 0.50, np.inf]
dti_labels = [
    'Target (<36%)',
    'Reasonable (36-41%)',
    'Less Favorable (41-43%)',
    'High (43-50%)',
    'Very High (>50%)'
]
df_cleaned['DTI_Band'] = create_bins(df_cleaned['DTI'], dti_bins, dti_labels)

# Create DTI_Band_10pct: 10% intervals (0–10%, ..., >=100%)
dti10_bins = [0, 0.01, 0.03, 0.05, 0.07, 0.10, np.inf]  # chú ý đây là dạng tỷ lệ (0.01 = 1%)
dti10_labels = ["<1%","1–3%", "3–5%", "5–7%", "7–10%", ">10%"]

df_cleaned['DTI_Band_10pct'] = create_bins(df_cleaned['DTI'], dti10_bins, dti10_labels)

# Loan Band
loan_bins = [0, 1000, 5000, 10000, 15000, np.inf]
loan_labels = [
    'No debt',
    'Low (1K–5K)',
    'Moderate (5K–10K)',
    'High (10K–15K)',
    'Very High (>15K)',
]
df_cleaned['LoanAmount_Band'] = create_bins(df_cleaned['Student_Loan_Amt'], loan_bins, loan_labels)

# Asset Band
asset_bins = [0, 500, 1000, 1500, 2000, 2500, np.inf]
asset_labels = ['<500', '501-1000', '1001-1500', '1501-2000', '2001-2500' ,'>2500']
df_cleaned['Asset_Band'] = create_bins(df_cleaned['Checking_Asset'], asset_bins, asset_labels)

def move_column(col_list, col_name, before_col):
    if col_name in col_list and before_col in col_list:
        col_list.remove(col_name)
        idx = col_list.index(before_col)
        col_list.insert(idx, col_name)
    return col_list

column_order = df_cleaned.columns.tolist()
column_order = move_column(column_order, 'FICO', 'FICO_Band')
column_order = move_column(column_order, 'DTI_Band', 'DTI_Band_10pct')
df_cleaned = df_cleaned[column_order]

# Assuming you have predicted_chargeoff as a numeric probability or binary prediction
# Example: if predicted_chargeoff is probability between 0 and 1
#chargeoff_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
#chargeoff_labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
#df_cleaned['Predicted_Chargeoff_Band'] = create_bins(df_cleaned['Ever_ChargeOff'], chargeoff_bins, chargeoff_labels)

# Copy predicted score as a raw column
df_cleaned['Predicted_Chargeoff_Score'] = df_cleaned['Ever_ChargeOff']

# Move it next to Predicted_Chargeoff_Band
column_order = df_cleaned.columns.tolist()
column_order = move_column(column_order, 'Predicted_Chargeoff_Score', 'Predicted_Chargeoff_Band')
df_cleaned = df_cleaned[column_order]

# 4. Export the DataFrame with new columns to CSV
output_path = '/content/sample_data/NTC_Consumer_Credit_Data.csv'
df_cleaned.to_csv(output_path, index=False)

print(f"\nData exported successfully with new columns and bands to {output_path}")
