import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb


#Examaning a correlation matrix of all the features
df = pd.read_csv('data/PCOS_Cleaned_Data.csv')

# Separate features and target
X = df.drop(["PCOS (Y/N)", "Patient File No."], axis=1)  # Drop non-predictive columns
y = df["PCOS (Y/N)"]

# For numerical features: ANOVA F-test
selector_num = SelectKBest(score_func=f_classif, k='all')

selector_num.fit(X.select_dtypes(include=['int64', 'float64']), y)

anova_f_scores = pd.DataFrame({
    'Feature': X.select_dtypes(include=['int64', 'float64']).columns,
    'ANOVA Score': selector_num.scores_
}).sort_values(by='ANOVA Score', ascending=False)

anova_p_values = pd.DataFrame({
    'Feature': X.select_dtypes(include=['int64', 'float64']).columns,
    'ANOVA Score': selector_num.pvalues_
}).sort_values(by='ANOVA Score', ascending=True)

print("Top Numerical Features by F:\n", anova_f_scores.head(10))
print("Top Numerical Features by P:\n", anova_p_values.head(10))


#test model training with p_values > 0.05 removed
selector = SelectKBest(score_func=f_classif, k=15)  # Select top 15 features
X_selected = selector.fit_transform(X, y)

# Get selected feature names

selected_features = X[['Follicle No. (R)', 'Follicle No. (L)', 'Skin darkening (Y/N)', 'hair growth(Y/N)', 'Weight gain(Y/N)', 'Cycle(R/I)', 'Fast food (Y/N)', 'Pimples(Y/N)', 'AMH(ng/mL)']]
X_train, X_test, y_train, y_test = train_test_split(
    selected_features, y, test_size=0.2, stratify=y, random_state=42
)

scale_pos_weight = np.sum(y == 0) / np.sum(y == 1)  # Adjust for imbalance

# Train XGBoost
# -------------
XGB_model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    early_stopping_rounds=20,
    random_state=42,
    eval_metric='auc'
)

# Early stopping to prevent overfitting
eval_set = [(X_train, y_train), (X_test, y_test)]
XGB_model.fit(
    X_train, y_train,
    eval_set=eval_set,
    verbose=True
)

# Evaluate
# --------
y_pred = XGB_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAUC-ROC:", roc_auc_score(y_test, y_pred))
