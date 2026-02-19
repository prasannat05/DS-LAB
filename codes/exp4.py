import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("/content/drive/MyDrive/loan_data.csv")
df = df.sample(200000, random_state=42)

# -----------------------------
# Encode Categorical Columns
# -----------------------------
cat_cols = ["purpose", "home_ownership_n", "addr_state"]
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# -----------------------------
# Features & Target
# -----------------------------
X = df.drop("non_repayment_status", axis=1)
y = df["non_repayment_status"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# SMOTE Oversampling
# -----------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", pd.Series(y_train_res).value_counts())

# -----------------------------
# Models
# -----------------------------
models = {
    "Logistic": LogisticRegression(max_iter=1000, n_jobs=-1),
    "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier()
}

results = []

# -----------------------------
# Train & Evaluate
# -----------------------------
for name, model in models.items():
    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    print(f"\n{name}")
    print("Accuracy:", acc)
    print("ROC-AUC:", roc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    results.append([name, acc, roc])

# -----------------------------
# Comparison Table
# -----------------------------
df_results = pd.DataFrame(results, columns=["Model", "Accuracy", "ROC-AUC"])
print("\nModel Comparison:\n", df_results)
