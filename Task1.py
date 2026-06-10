# ==========================================
# Employee Financial Stability Prediction
# ML Project (Classification)
# ==========================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# ----------------------------
# STEP 1: Load Dataset
# ----------------------------

data = pd.read_csv("employee_financial_data.csv")

print("\nDataset Loaded Successfully\n")
print(data.head())


# ----------------------------
# STEP 2: Drop useless columns
# ----------------------------

data.drop(columns=["Employee_ID", "Name"], inplace=True, errors="ignore")


# ----------------------------
# STEP 3: Encode categorical data
# ----------------------------

le = LabelEncoder()

for col in data.columns:
    if data[col].dtype == "object":
        data[col] = le.fit_transform(data[col])


# ----------------------------
# STEP 4: Feature Engineering
# ----------------------------

data["Savings_Ratio"] = data["Savings_Amount"] / (data["Net_Salary"] + 1)
data["Expense_Ratio"] = data["Total_Expenditure"] / (data["Net_Salary"] + 1)
data["Loan_Burden"] = data["EMI_or_Loan_Payment"] / (data["Net_Salary"] + 1)
data["Disposable_Income"] = data["Net_Salary"] - data["Total_Expenditure"]


print("\nFeature Engineering Done\n")


# ----------------------------
# STEP 5: Define Features & Target
# ----------------------------

features = [
    "Monthly_Salary",
    "Net_Salary",
    "Income_Tax",
    "PF_Contribution",
    "Insurance_Deduction",
    "Other_Deductions",
    "Rent_Expense",
    "Grocery_Expense",
    "EMI_or_Loan_Payment",
    "Entertainment_Expense",
    "Other_Expenses",
    "Savings_Amount",
    "Investments",
    "Total_Expenditure",
    "Profit",
    "Savings_Ratio",
    "Expense_Ratio",
    "Loan_Burden",
    "Disposable_Income"
]

X = data[features]
y = data["Class"]


# ----------------------------
# STEP 6: Train-Test Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ----------------------------
# STEP 7: Train Model
# ----------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Completed\n")


# ----------------------------
# STEP 8: Predictions
# ----------------------------

y_pred = model.predict(X_test)


# ----------------------------
# STEP 9: Evaluation
# ----------------------------

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ----------------------------
# STEP 10: Manual Prediction
# ----------------------------

print("\n===== TEST NEW EMPLOYEE =====")

input_data = {
    "Monthly_Salary": float(input("Monthly Salary: ")),
    "Net_Salary": float(input("Net Salary: ")),
    "Income_Tax": float(input("Income Tax: ")),
    "PF_Contribution": float(input("PF Contribution: ")),
    "Insurance_Deduction": float(input("Insurance Deduction: ")),
    "Other_Deductions": float(input("Other Deductions: ")),
    "Rent_Expense": float(input("Rent Expense: ")),
    "Grocery_Expense": float(input("Grocery Expense: ")),
    "EMI_or_Loan_Payment": float(input("EMI/Loan Payment: ")),
    "Entertainment_Expense": float(input("Entertainment Expense: ")),
    "Other_Expenses": float(input("Other Expenses: ")),
    "Savings_Amount": float(input("Savings Amount: ")),
    "Investments": float(input("Investments: ")),
    "Total_Expenditure": float(input("Total Expenditure: ")),
    "Profit": float(input("Profit: "))
}

# Feature engineering for input
input_data["Savings_Ratio"] = input_data["Savings_Amount"] / (input_data["Net_Salary"] + 1)
input_data["Expense_Ratio"] = input_data["Total_Expenditure"] / (input_data["Net_Salary"] + 1)
input_data["Loan_Burden"] = input_data["EMI_or_Loan_Payment"] / (input_data["Net_Salary"] + 1)
input_data["Disposable_Income"] = input_data["Net_Salary"] - input_data["Total_Expenditure"]


# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Match feature order
input_df = input_df[X.columns]

# Predict
prediction = model.predict(input_df)

print("\nPrediction Result:", prediction[0])