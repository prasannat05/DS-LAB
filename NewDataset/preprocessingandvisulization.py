import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("/content/drive/MyDrive/DS/DATASET/split_files/loan_part_13.csv")

df.drop(columns=["id", "zip_code", "title", "desc"], inplace=True)

df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")
df["issue_year"] = df["issue_d"].dt.year
df.drop(columns=["issue_d"], inplace=True)


categorical_cols = ["emp_length", "purpose", "home_ownership_n", "addr_state"]
df[categorical_cols] = df[categorical_cols].fillna("Unknown")


emp_map = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
    "4 years": 4, "5 years": 5, "6 years": 6,
    "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10, "NI": -1
}
df["emp_length_n"] = df["emp_length"].map(emp_map)\


df.to_csv("loan_cleaned.csv", index=False)

plt.figure()
df["Default"].value_counts().plot(kind="bar")
plt.xlabel("Default (0 = Paid, 1 = Charged Off)")
plt.ylabel("Count")
plt.title("Loan Default Class Distribution")
plt.show()

plt.figure()
df.boxplot(column="loan_amnt", by="Default")
plt.title("Loan amount vs Default")
plt.xlabel("Default")
plt.ylabel("Loan Amount")
plt.suptitle("")
plt.show()

plt.figure()
df.groupby("purpose")["Default"].mean().sort_values().plot(kind="bar")
plt.ylabel("Default Rate")
plt.title("Default Rate by Loan Purpose")
plt.show()

plt.figure()
df.groupby("home_ownership_n")["Default"].mean().plot(kind="bar")
plt.ylabel("Default Rate")
plt.title("Default Rate by Home Ownership")
plt.show()

corr_features = [
    "loan_amnt",
    "revenue",
    "fico_n",
    "dti_n",
    "emp_length_n",
    "Default"
]

corr = df[corr_features].corr()

plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=45)
plt.yticks(range(len(corr)), corr.columns)
plt.title("Correlation Heatmap")
plt.show()


df["loan_bin"] = pd.cut(df["loan_amnt"], bins=10)

loan_default = df.groupby("loan_bin")["Default"].mean()

plt.figure()
loan_default.plot(marker="o")
plt.xlabel("Loan Amount Range")
plt.ylabel("Default Rate")
plt.title("Default Rate vs Loan Amount")
plt.xticks(rotation=45)
plt.show()

df["fico_bin"] = pd.cut(df["fico_n"], bins=10)

fico_default = df.groupby("fico_bin")["Default"].mean()

plt.figure()
fico_default.plot(marker="o")
plt.xlabel("FICO Score Range")
plt.ylabel("Default Rate")
plt.title("Default Rate vs FICO Score")
plt.xticks(rotation=45)
plt.show()


