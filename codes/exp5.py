#EXP 5
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.feature_selection import (SelectKBest, f_regression, f_classif,
    RFE, SelectFromModel, mutual_info_regression, mutual_info_classif)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/content/drive/MyDrive/DS DATA SET/updated_splited.csv')

# Encode categorical
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# ============================================================================
# LINEAR REGRESSION
# ============================================================================

X_linear = df.drop(['loan_amnt', 'loan_repaid'], axis=1)
y_linear = df['loan_amnt']

X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
    X_linear, y_linear, test_size=0.2, random_state=42)

scaler_lr = StandardScaler()
X_train_lr_scaled = scaler_lr.fit_transform(X_train_lr)
X_test_lr_scaled = scaler_lr.transform(X_test_lr)

results_lr = {}

# All Features
lr = LinearRegression().fit(X_train_lr_scaled, y_train_lr)
y_pred = lr.predict(X_test_lr_scaled)
results_lr['All Features'] = {
    'n_features': X_linear.shape[1], 'r2': r2_score(y_test_lr, y_pred),
    'rmse': np.sqrt(mean_squared_error(y_test_lr, y_pred)),
    'mae': mean_absolute_error(y_test_lr, y_pred),
    'features': list(X_linear.columns)}

# SelectKBest F-stat
kb = SelectKBest(f_regression, k=5).fit(X_train_lr_scaled, y_train_lr)
lr = LinearRegression().fit(kb.transform(X_train_lr_scaled), y_train_lr)
y_pred = lr.predict(kb.transform(X_test_lr_scaled))
results_lr['SelectKBest (F-stat)'] = {
    'n_features': 5, 'r2': r2_score(y_test_lr, y_pred),
    'rmse': np.sqrt(mean_squared_error(y_test_lr, y_pred)),
    'mae': mean_absolute_error(y_test_lr, y_pred),
    'features': X_linear.columns[kb.get_support()].tolist()}

# SelectKBest MI
mi = SelectKBest(mutual_info_regression, k=5).fit(X_train_lr_scaled, y_train_lr)
lr = LinearRegression().fit(mi.transform(X_train_lr_scaled), y_train_lr)
y_pred = lr.predict(mi.transform(X_test_lr_scaled))
results_lr['SelectKBest (MI)'] = {
    'n_features': 5, 'r2': r2_score(y_test_lr, y_pred),
    'rmse': np.sqrt(mean_squared_error(y_test_lr, y_pred)),
    'mae': mean_absolute_error(y_test_lr, y_pred),
    'features': X_linear.columns[mi.get_support()].tolist()}

# RFE
rfe = RFE(LinearRegression(), n_features_to_select=5).fit(X_train_lr_scaled, y_train_lr)
lr = LinearRegression().fit(rfe.transform(X_train_lr_scaled), y_train_lr)
y_pred = lr.predict(rfe.transform(X_test_lr_scaled))
results_lr['RFE'] = {
    'n_features': 5, 'r2': r2_score(y_test_lr, y_pred),
    'rmse': np.sqrt(mean_squared_error(y_test_lr, y_pred)),
    'mae': mean_absolute_error(y_test_lr, y_pred),
    'features': X_linear.columns[rfe.get_support()].tolist()}

# SelectFromModel
sfm = SelectFromModel(RandomForestRegressor(50, random_state=42), threshold='median')
sfm.fit(X_train_lr, y_train_lr)
X_train_sfm = StandardScaler().fit_transform(sfm.transform(X_train_lr))
X_test_sfm = StandardScaler().fit(sfm.transform(X_train_lr)).transform(sfm.transform(X_test_lr))
lr = LinearRegression().fit(X_train_sfm, y_train_lr)
y_pred = lr.predict(X_test_sfm)
results_lr['SelectFromModel'] = {
    'n_features': sfm.get_support().sum(), 'r2': r2_score(y_test_lr, y_pred),
    'rmse': np.sqrt(mean_squared_error(y_test_lr, y_pred)),
    'mae': mean_absolute_error(y_test_lr, y_pred),
    'features': X_linear.columns[sfm.get_support()].tolist()}

# Lasso (L1) — zero coefficients = feature elimination
lasso = Lasso(alpha=0.01, random_state=42).fit(X_train_lr_scaled, y_train_lr)
lasso_support = np.abs(lasso.coef_) > 0
y_pred = lasso.predict(X_test_lr_scaled)
results_lr['Lasso (L1)'] = {
    'n_features': lasso_support.sum(), 'r2': r2_score(y_test_lr, y_pred),
    'rmse': np.sqrt(mean_squared_error(y_test_lr, y_pred)),
    'mae': mean_absolute_error(y_test_lr, y_pred),
    'features': X_linear.columns[lasso_support].tolist()}

# Ridge (L2) — shrinks all, top 5 by coefficient magnitude
ridge = Ridge(alpha=1.0).fit(X_train_lr_scaled, y_train_lr)
top5_ridge = np.argsort(np.abs(ridge.coef_))[-5:]
ridge_support = np.zeros(X_linear.shape[1], dtype=bool)
ridge_support[top5_ridge] = True
lr_ridge = LinearRegression().fit(X_train_lr_scaled[:, top5_ridge], y_train_lr)
y_pred = lr_ridge.predict(X_test_lr_scaled[:, top5_ridge])
results_lr['Ridge (L2) Top5'] = {
    'n_features': 5, 'r2': r2_score(y_test_lr, y_pred),
    'rmse': np.sqrt(mean_squared_error(y_test_lr, y_pred)),
    'mae': mean_absolute_error(y_test_lr, y_pred),
    'features': X_linear.columns[top5_ridge].tolist()}

print("="*80)
print("LINEAR REGRESSION RESULTS")
print("="*80)
print(f"{'Method':<25} {'Features':<10} {'R²':<10} {'RMSE':<12} {'MAE':<12}")
print("-"*80)
for method, metrics in results_lr.items():
    print(f"{method:<25} {metrics['n_features']:<10} {metrics['r2']:<10.4f} "
          f"{metrics['rmse']:<11,.0f} {metrics['mae']:<11,.0f}")
print("\n" + "-"*80)
print("SELECTED FEATURES")
print("-"*80)
for method, metrics in results_lr.items():
    print(f"{method}: {', '.join(metrics['features'])}")

best_lr = max(results_lr.items(), key=lambda x: x[1]['r2'])
print(f"\nBest: {best_lr[0]} (R² = {best_lr[1]['r2']:.4f})")

# ============================================================================
# LOGISTIC REGRESSION
# ============================================================================

X_logistic = df.drop('loan_repaid', axis=1)
y_logistic = df['loan_repaid']

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
    X_logistic, y_logistic, test_size=0.2, random_state=42, stratify=y_logistic)

scaler_log = StandardScaler()
X_train_log_scaled = scaler_log.fit_transform(X_train_log)
X_test_log_scaled = scaler_log.transform(X_test_log)

X_train_smote, y_train_smote = SMOTE(random_state=42).fit_resample(X_train_log_scaled, y_train_log)

def get_threshold(y_true, y_proba):
    best_f1, best_th = 0, 0.5
    for th in np.arange(0.3, 0.7, 0.05):
        f1 = f1_score(y_true, (y_proba >= th).astype(int))
        if f1 > best_f1:
            best_f1, best_th = f1, th
    return best_th

results_log = {}

# All Features
log = LogisticRegression(max_iter=1000, C=0.1, random_state=42).fit(X_train_smote, y_train_smote)
y_proba = log.predict_proba(X_test_log_scaled)[:, 1]
threshold = get_threshold(y_test_log, y_proba)
y_pred = (y_proba >= threshold).astype(int)
results_log['All Features'] = {
    'n_features': X_logistic.shape[1], 'roc_auc': roc_auc_score(y_test_log, y_proba),
    'accuracy': accuracy_score(y_test_log, y_pred), 'precision': precision_score(y_test_log, y_pred),
    'recall': recall_score(y_test_log, y_pred), 'f1': f1_score(y_test_log, y_pred),
    'features': list(X_logistic.columns)}

# SelectKBest F-stat
kb_log = SelectKBest(f_classif, k=5).fit(X_train_smote, y_train_smote)
log = LogisticRegression(max_iter=1000, C=0.1, random_state=42).fit(
    kb_log.transform(X_train_smote), y_train_smote)
y_proba = log.predict_proba(kb_log.transform(X_test_log_scaled))[:, 1]
y_pred = (y_proba >= threshold).astype(int)
results_log['SelectKBest (F-stat)'] = {
    'n_features': 5, 'roc_auc': roc_auc_score(y_test_log, y_proba),
    'accuracy': accuracy_score(y_test_log, y_pred), 'precision': precision_score(y_test_log, y_pred),
    'recall': recall_score(y_test_log, y_pred), 'f1': f1_score(y_test_log, y_pred),
    'features': X_logistic.columns[kb_log.get_support()].tolist()}

# SelectKBest MI
mi_log = SelectKBest(mutual_info_classif, k=5).fit(X_train_smote, y_train_smote)
log = LogisticRegression(max_iter=1000, C=0.1, random_state=42).fit(
    mi_log.transform(X_train_smote), y_train_smote)
y_proba = log.predict_proba(mi_log.transform(X_test_log_scaled))[:, 1]
y_pred = (y_proba >= threshold).astype(int)
results_log['SelectKBest (MI)'] = {
    'n_features': 5, 'roc_auc': roc_auc_score(y_test_log, y_proba),
    'accuracy': accuracy_score(y_test_log, y_pred), 'precision': precision_score(y_test_log, y_pred),
    'recall': recall_score(y_test_log, y_pred), 'f1': f1_score(y_test_log, y_pred),
    'features': X_logistic.columns[mi_log.get_support()].tolist()}

# RFE
rfe_log = RFE(LogisticRegression(max_iter=1000, random_state=42), n_features_to_select=5)
rfe_log.fit(X_train_smote, y_train_smote)
log = LogisticRegression(max_iter=1000, C=0.1, random_state=42).fit(
    rfe_log.transform(X_train_smote), y_train_smote)
y_proba = log.predict_proba(rfe_log.transform(X_test_log_scaled))[:, 1]
y_pred = (y_proba >= threshold).astype(int)
results_log['RFE'] = {
    'n_features': 5, 'roc_auc': roc_auc_score(y_test_log, y_proba),
    'accuracy': accuracy_score(y_test_log, y_pred), 'precision': precision_score(y_test_log, y_pred),
    'recall': recall_score(y_test_log, y_pred), 'f1': f1_score(y_test_log, y_pred),
    'features': X_logistic.columns[rfe_log.get_support()].tolist()}

# SelectFromModel
sfm_log = SelectFromModel(RandomForestClassifier(50, random_state=42, class_weight='balanced'),
                          threshold='median')
sfm_log.fit(X_train_log_scaled, y_train_log)
X_train_sfm_log, y_train_sfm_log = SMOTE(random_state=42).fit_resample(
    sfm_log.transform(X_train_log_scaled), y_train_log)
log = LogisticRegression(max_iter=1000, C=0.1, random_state=42).fit(X_train_sfm_log, y_train_sfm_log)
y_proba = log.predict_proba(sfm_log.transform(X_test_log_scaled))[:, 1]
y_pred = (y_proba >= threshold).astype(int)
results_log['SelectFromModel'] = {
    'n_features': sfm_log.get_support().sum(), 'roc_auc': roc_auc_score(y_test_log, y_proba),
    'accuracy': accuracy_score(y_test_log, y_pred), 'precision': precision_score(y_test_log, y_pred),
    'recall': recall_score(y_test_log, y_pred), 'f1': f1_score(y_test_log, y_pred),
    'features': X_logistic.columns[sfm_log.get_support()].tolist()}

# L1 Logistic — sparse solution, zero coefs = eliminated features
log_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
log_l1.fit(X_train_smote, y_train_smote)
l1_support = np.abs(log_l1.coef_[0]) > 0
y_proba = log_l1.predict_proba(X_test_log_scaled)[:, 1]
threshold_l1 = get_threshold(y_test_log, y_proba)
y_pred = (y_proba >= threshold_l1).astype(int)
results_log['L1 Logistic'] = {
    'n_features': l1_support.sum(), 'roc_auc': roc_auc_score(y_test_log, y_proba),
    'accuracy': accuracy_score(y_test_log, y_pred), 'precision': precision_score(y_test_log, y_pred),
    'recall': recall_score(y_test_log, y_pred), 'f1': f1_score(y_test_log, y_pred),
    'features': X_logistic.columns[l1_support].tolist()}

# L2 Logistic — shrinks all, top 5 by coefficient magnitude
log_l2 = LogisticRegression(penalty='l2', solver='lbfgs', C=0.1, max_iter=1000, random_state=42)
log_l2.fit(X_train_smote, y_train_smote)
top5_l2 = np.argsort(np.abs(log_l2.coef_[0]))[-5:]
log_l2_top5 = LogisticRegression(max_iter=1000, C=0.1, random_state=42).fit(
    X_train_smote[:, top5_l2], y_train_smote)
y_proba = log_l2_top5.predict_proba(X_test_log_scaled[:, top5_l2])[:, 1]
threshold_l2 = get_threshold(y_test_log, y_proba)
y_pred = (y_proba >= threshold_l2).astype(int)
results_log['L2 Logistic Top5'] = {
    'n_features': 5, 'roc_auc': roc_auc_score(y_test_log, y_proba),
    'accuracy': accuracy_score(y_test_log, y_pred), 'precision': precision_score(y_test_log, y_pred),
    'recall': recall_score(y_test_log, y_pred), 'f1': f1_score(y_test_log, y_pred),
    'features': X_logistic.columns[top5_l2].tolist()}

print("\n" + "="*80)
print("LOGISTIC REGRESSION RESULTS")
print("="*80)
print(f"{'Method':<25} {'Features':<10} {'ROC-AUC':<10} {'Accuracy':<10} {'Precision':<11} {'Recall':<10} {'F1':<10}")
print("-"*80)
for method, metrics in results_log.items():
    print(f"{method:<25} {metrics['n_features']:<10} {metrics['roc_auc']:<10.4f} "
          f"{metrics['accuracy']:<10.4f} {metrics['precision']:<11.4f} "
          f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f}")
print("\n" + "-"*80)
print("SELECTED FEATURES")
print("-"*80)
for method, metrics in results_log.items():
    print(f"{method}: {', '.join(metrics['features'])}")

best_log = max(results_log.items(), key=lambda x: x[1]['f1'])
print(f"\nBest: {best_log[0]} (F1 = {best_log[1]['f1']:.4f})")
