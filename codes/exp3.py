import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("/content/drive/MyDrive/DS DATA SET/loan_data.csv")
num_cols = [
    "revenue",
    "dti_n",
    "loan_amnt",
    "fico_n",
    "experience_c",
    "non_repayment_status",
    "issue_year",
    "emp_length_n"
]
df_num = df[num_cols]
cov_matrix = df_num.cov()
corr_matrix = df_num.corr()
print("Covariance Matrix:")
print(cov_matrix)
print("\nCorrelation Matrix:")
print(corr_matrix)
plt.figure(figsize=(10, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)
plt.title("Correlation Heatmap of Loan Features")
plt.show()
plt.figure()
plt.scatter(df["dti_n"], df["non_repayment_status"])
plt.xlabel("DTI Ratio")
plt.ylabel("Non-Repayment Status")
plt.title("DTI vs Non-Repayment")
plt.show()
plt.figure()
plt.scatter(df["fico_n"], df["non_repayment_status"])
plt.xlabel("FICO Score")
plt.ylabel("Non-Repayment Status")
plt.title("FICO Score vs Non-Repayment")
plt.show()
plt.figure()
plt.scatter(df["loan_amnt"], df["revenue"])
plt.xlabel("Loan Amount")
plt.ylabel("Revenue")
plt.title("Loan Amount vs Revenue")
plt.show()
