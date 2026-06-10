import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------
# 1. Load dataset
# ----------------------------

data = fetch_openml("diabetes", version=1, as_frame=True)

df = data.frame

print("\nDataset Loaded\n")
print(df.head())


# ----------------------------
# 2. Features & Target
# ----------------------------

X = df.drop("class", axis=1)
y = df["class"]

# convert target to numeric
y = y.map({"tested_positive": 1, "tested_negative": 0})


# ----------------------------
# 3. Train-test split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ----------------------------
# 4. Model (Random Forest)
# ----------------------------

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)


# ----------------------------
# 5. Predictions
# ----------------------------

y_pred = model.predict(X_test)


# ----------------------------
# 6. Evaluation
# ----------------------------

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))


# ----------------------------
# 7. Predict new patient
# ----------------------------

print("\n--- NEW PATIENT PREDICTION ---")

input_data = {
    "preg": float(input("Pregnancies: ")),
    "plas": float(input("Glucose: ")),
    "pres": float(input("Blood Pressure: ")),
    "skin": float(input("Skin Thickness: ")),
    "insu": float(input("Insulin: ")),
    "mass": float(input("BMI: ")),
    "pedi": float(input("Diabetes Pedigree Function: ")),
    "age": float(input("Age: "))
}

input_df = pd.DataFrame([input_data])

prediction = model.predict(input_df)

if prediction[0] == 1:
    print("\nResult: DIABETES DETECTED")
else:
    print("\nResult: NO DIABETES")