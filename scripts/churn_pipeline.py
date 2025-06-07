import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# SECTION 1 – Data Cleaning

# Load the dataset from the data folder
df = pd.read_csv("data/telco_churn.csv")

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Strip whitespace from string values
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Convert 'TotalCharges' to float, replacing non-convertible entries with NaN
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Drop rows where 'TotalCharges' is missing (optional)
df = df.dropna(subset=["TotalCharges"])
df = df.reset_index(drop=True)

# Convert binary "Yes"/"No" columns to boolean
binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
for col in binary_cols:
    df[col] = df[col].map({"Yes": True, "No": False})

# Replace "No phone service" with "No" in 'MultipleLines'
if "MultipleLines" in df.columns:
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")

# Replace "No internet service" with "No" in internet-related service columns
internet_cols = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]
for col in internet_cols:
    df[col] = df[col].replace("No internet service", "No")

# Convert 'SeniorCitizen' from integer to boolean
df["SeniorCitizen"] = df["SeniorCitizen"].astype(bool)

# Check for remaining missing values
print(df.isnull().sum())

# Export cleaned dataframe for analysis
df.to_csv("data/telco_churn_clean.csv", index=False)

# SECTION 2 – Univariate Analysis

# SECTION 2.1 – Univariate Analysis: Categorical Variables

# Identify categorical columns (excluding ID)
categorical_cols = df.select_dtypes(include="object").columns.tolist()
if "customerID" in categorical_cols:
    categorical_cols.remove("customerID")

print("\n--- SECTION 2.1 Descriptive Statistics for Categorical Variables ---")
# Display value counts and percentages for each categorical variable
for col in categorical_cols:
    print(f"\n--- {col} ---")
    print(df[col].value_counts())
    print("\nPercentages:")
    print(round(df[col].value_counts(normalize=True) * 100, 2))

# SECTION 2.2 – Univariate Analysis: Numerical Variables

numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

print("\n--- SECTION 2.2 – Descriptive Statistics for Numerical Variables ---")
print(df[numerical_cols].describe())

print("\n--- Mean vs Median Comparison ---")
for col in numerical_cols:
    mean = df[col].mean()
    median = df[col].median()
    print(f"{col}: mean = {round(mean, 2)}, median = {round(median, 2)}")

# Create a summary statistics table for numerical variables
summary_stats = df[numerical_cols].agg(["mean", "median", "std", "min", "max"]).T
summary_stats = summary_stats.rename(columns={"mean": "Mean", "median": "Median", "std": "Std Dev", "min": "Min", "max": "Max"})
print("\n--- Summary Statistics Table ---")
print(summary_stats)

# SECTION 2.3 – Categorical Variables vs Churn

print("\n--- SECTION 2.3 – Categorical Variables vs Churn ---")

# Grouped bar plots for each categorical variable vs churn
for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col, hue="Churn")
    plt.title(f"Churn by {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Calculate churn percentage per category for each categorical variable
print("\n--- Churn Rate per Category ---")
for col in categorical_cols:
    churn_rate = df.groupby(col)["Churn"].mean().round(2) * 100
    print(f"\n{col}:\n{churn_rate}")


# SECTION 2.4 – Numerical Variables vs Churn
print("\n--- SECTION 2.4 – Numerical Variables vs Churn ---")

# Create boxplots of numerical variables grouped by Churn
for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="Churn", y=col)
    plt.title(f"{col} Distribution by Churn")
    plt.tight_layout()
    plt.show()
    grouped_stats = df.groupby("Churn")[col].agg(["mean", "median", "std"]).round(2)
    print(f"\n{col} grouped by Churn:\n{grouped_stats}")


# SECTION 2.5 – Distribution of Numerical Variables
print("\n--- SECTION 2.5 – Distribution of Numerical Variables ---")

# Histogram and KDE for each numeric variable
for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Print binned value counts
    print(f"\n--- Value Ranges for {col} ---")
    print(pd.cut(df[col], bins=5).value_counts().sort_index())

# SECTION 3.1 – Correlation Analysis of Numerical Variables
print("\n--- SECTION 3.1 – Correlation Analysis of Numerical Variables ---")

# Select numerical features for correlation
corr_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
# Include Churn as numeric
df_corr = df[corr_cols + ["Churn"]].copy()
df_corr["Churn"] = df_corr["Churn"].astype(int)

# Compute correlation matrix
correlation_matrix = df_corr.corr()

# Display the correlation matrix
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Matrix of Numerical Variables")
plt.tight_layout()
plt.show()

# SECTION 3.2 – Multivariate Visual Analysis
print("\n--- SECTION 3.2 – Multivariate Visual Analysis ---")

# Boxplot: Contract vs Tenure with Churn as hue
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="Contract", y="tenure", hue="Churn")
plt.title("Tenure by Contract Type and Churn")
plt.xlabel("Contract Type")
plt.ylabel("Tenure (months)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Scatterplot: MonthlyCharges vs TotalCharges colored by Churn
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="MonthlyCharges", y="TotalCharges", hue="Churn", alpha=0.6)
plt.title("Monthly Charges vs Total Charges by Churn")
plt.xlabel("Monthly Charges")
plt.ylabel("Total Charges")
plt.tight_layout()
plt.show()

# Summary statistics: Tenure by Contract and Churn
print("\n--- Summary Statistics: Tenure by Contract and Churn ---")
tenure_summary = df.groupby(["Contract", "Churn"])["tenure"].agg(["mean", "median", "std", "count"]).round(2)
print(tenure_summary)

# Summary statistics: MonthlyCharges and TotalCharges by Churn
print("\n--- Summary Statistics: Charges by Churn ---")
charges_summary = df.groupby("Churn")[["MonthlyCharges", "TotalCharges"]].agg(["mean", "median", "std", "count"]).round(2)
print(charges_summary)


# SECTION 1 – Data Cleaning

# Load the dataset from the data folder
df = pd.read_csv("data/telco_churn.csv")

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Strip whitespace from string values
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Convert 'TotalCharges' to float, replacing non-convertible entries with NaN
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Drop rows where 'TotalCharges' is missing (optional)
df = df.dropna(subset=["TotalCharges"])
df = df.reset_index(drop=True)

# Convert binary "Yes"/"No" columns to boolean
binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
for col in binary_cols:
    df[col] = df[col].map({"Yes": True, "No": False})

# Replace "No phone service" with "No" in 'MultipleLines'
if "MultipleLines" in df.columns:
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")

# Replace "No internet service" with "No" in internet-related service columns
internet_cols = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]
for col in internet_cols:
    df[col] = df[col].replace("No internet service", "No")

# Convert 'SeniorCitizen' from integer to boolean
df["SeniorCitizen"] = df["SeniorCitizen"].astype(bool)

# Check for remaining missing values
print(df.isnull().sum())

# Export cleaned dataframe for analysis
df.to_csv("data/telco_churn_clean.csv", index=False)

# SECTION 2 – Univariate Analysis

# SECTION 2.1 – Univariate Analysis: Categorical Variables

# Identify categorical columns (excluding ID)
categorical_cols = df.select_dtypes(include="object").columns.tolist()
if "customerID" in categorical_cols:
    categorical_cols.remove("customerID")

print("\n--- SECTION 2.1 Descriptive Statistics for Categorical Variables ---")
# Display value counts and percentages for each categorical variable
for col in categorical_cols:
    print(f"\n--- {col} ---")
    print(df[col].value_counts())
    print("\nPercentages:")
    print(round(df[col].value_counts(normalize=True) * 100, 2))

# SECTION 2.2 – Univariate Analysis: Numerical Variables

numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

print("\n--- SECTION 2.2 – Descriptive Statistics for Numerical Variables ---")
print(df[numerical_cols].describe())

print("\n--- Mean vs Median Comparison ---")
for col in numerical_cols:
    mean = df[col].mean()
    median = df[col].median()
    print(f"{col}: mean = {round(mean, 2)}, median = {round(median, 2)}")

# Create a summary statistics table for numerical variables
summary_stats = df[numerical_cols].agg(["mean", "median", "std", "min", "max"]).T
summary_stats = summary_stats.rename(columns={"mean": "Mean", "median": "Median", "std": "Std Dev", "min": "Min", "max": "Max"})
print("\n--- Summary Statistics Table ---")
print(summary_stats)

# SECTION 2.3 – Categorical Variables vs Churn

print("\n--- SECTION 2.3 – Categorical Variables vs Churn ---")

# Grouped bar plots for each categorical variable vs churn
for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col, hue="Churn")
    plt.title(f"Churn by {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Calculate churn percentage per category for each categorical variable
print("\n--- Churn Rate per Category ---")
for col in categorical_cols:
    churn_rate = df.groupby(col)["Churn"].mean().round(2) * 100
    print(f"\n{col}:\n{churn_rate}")


# SECTION 2.4 – Numerical Variables vs Churn
print("\n--- SECTION 2.4 – Numerical Variables vs Churn ---")

# Create boxplots of numerical variables grouped by Churn
for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="Churn", y=col)
    plt.title(f"{col} Distribution by Churn")
    plt.tight_layout()
    plt.show()
    grouped_stats = df.groupby("Churn")[col].agg(["mean", "median", "std"]).round(2)
    print(f"\n{col} grouped by Churn:\n{grouped_stats}")


# SECTION 2.5 – Distribution of Numerical Variables
print("\n--- SECTION 2.5 – Distribution of Numerical Variables ---")

# Histogram and KDE for each numeric variable
for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Print binned value counts
    print(f"\n--- Value Ranges for {col} ---")
    print(pd.cut(df[col], bins=5).value_counts().sort_index())

# SECTION 3.1 – Correlation Analysis of Numerical Variables
print("\n--- SECTION 3.1 – Correlation Analysis of Numerical Variables ---")

# Select numerical features for correlation
corr_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
# Include Churn as numeric
df_corr = df[corr_cols + ["Churn"]].copy()
df_corr["Churn"] = df_corr["Churn"].astype(int)

# Compute correlation matrix
correlation_matrix = df_corr.corr()

# Display the correlation matrix
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Matrix of Numerical Variables")
plt.tight_layout()
plt.show()

# SECTION 3.2 – Multivariate Visual Analysis
print("\n--- SECTION 3.2 – Multivariate Visual Analysis ---")

# Boxplot: Contract vs Tenure with Churn as hue
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="Contract", y="tenure", hue="Churn")
plt.title("Tenure by Contract Type and Churn")
plt.xlabel("Contract Type")
plt.ylabel("Tenure (months)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Scatterplot: MonthlyCharges vs TotalCharges colored by Churn
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="MonthlyCharges", y="TotalCharges", hue="Churn", alpha=0.6)
plt.title("Monthly Charges vs Total Charges by Churn")
plt.xlabel("Monthly Charges")
plt.ylabel("Total Charges")
plt.tight_layout()
plt.show()

# Summary statistics: Tenure by Contract and Churn
print("\n--- Summary Statistics: Tenure by Contract and Churn ---")
tenure_summary = df.groupby(["Contract", "Churn"])["tenure"].agg(["mean", "median", "std", "count"]).round(2)
print(tenure_summary)

# Summary statistics: MonthlyCharges and TotalCharges by Churn
print("\n--- Summary Statistics: Charges by Churn ---")
charges_summary = df.groupby("Churn")[["MonthlyCharges", "TotalCharges"]].agg(["mean", "median", "std", "count"]).round(2)
print(charges_summary)

# SECTION 3.3 – Exploratory Feature Engineering
print("\n--- SECTION 3.3 – Exploratory Feature Engineering ---")

# 1. IsNewCustomer: customers with tenure < 6 months
# Hypothesis: New customers are more likely to churn
df["IsNewCustomer"] = df["tenure"] < 6

# 2. IsLongTerm: customers on a One year or Two year contract
# Hypothesis: Long-term contracts reduce churn risk
df["IsLongTerm"] = df["Contract"].isin(["One year", "Two year"])

# 3. HasFiber: customer is subscribed to Fiber optic internet
# Hypothesis: Fiber users may be more demanding / have higher churn
df["HasFiber"] = df["InternetService"] == "Fiber optic"

# 4. IsTechProtected: customer has technical support enabled
# Hypothesis: Support users may experience fewer issues and churn less
df["IsTechProtected"] = df["TechSupport"] == "Yes"

# Display the first few rows of the new engineered features
print("\nSample of engineered features:\n", df[["IsNewCustomer", "IsLongTerm", "HasFiber", "IsTechProtected"]].head())

# Optional: Save updated dataframe with engineered features
df.to_csv("../data/telco_churn_enriched.csv", index=False)

# SECTION 4.1 – Feature Encoding
print("\n--- SECTION 4.1 – Feature Encoding ---")

# Encode the target variable
df["Churn"] = df["Churn"].astype(int)

# Identify categorical columns to encode (excluding engineered and boolean columns)
exclude_cols = [
    "customerID", "Churn", "IsNewCustomer", "IsLongTerm", "HasFiber", "IsTechProtected"
]
cat_cols = df.select_dtypes(include="object").columns.difference(exclude_cols).tolist()

# Apply one-hot encoding
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Export encoded data
df_encoded.to_csv("../data/telco_churn_encoded.csv", index=False)
print("Feature encoding complete. Encoded dataset saved as 'telco_churn_encoded.csv'")


# SECTION 4.2 – Model training and model evaluation
print("\n--- SECTION 4.2 – Model Training ---")


# Load encoded dataset
df_model = pd.read_csv("../data/telco_churn_encoded.csv")

# Separate features and target
X = df_model.drop(columns=["Churn", "customerID"], errors='ignore')
y = df_model["Churn"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
print("\nLogistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)


print("\nRandom Forest Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

#Decision Tree
dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("\nDecision Tree Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))

# Visualize Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=X.columns.tolist(), class_names=["No Churn", "Churn"],
          filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree Visualization (max_depth=4)")
plt.tight_layout()
plt.show()

# SECTION 4.3 – Feature Importance and Interpretation
print("\n--- SECTION 4.3 – Feature Importance and Interpretation ---")

# Create a dataframe of feature importances
importances = rf_model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Display top 15 features
print("\nTop 15 Most Important Features (Random Forest):")
print(importance_df.head(15))

# Plot top 15 feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df.head(15), palette="viridis")
plt.title("Top 15 Feature Importances (Random Forest)")
plt.tight_layout()
plt.show()