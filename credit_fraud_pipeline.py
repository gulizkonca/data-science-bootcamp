Python 3.13.2 (v3.13.2:4f8bb3947cf, Feb  4 2025, 11:51:10) [Clang 15.0.0 (clang-1500.3.9.4)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
#!/usr/bin/env python
# pip install pandas scikit-learn imbalanced-learn joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
... from imblearn.pipeline import Pipeline as ImbPipeline
... import joblib
... 
... DATA_PATH = "creditcard.csv"  # Kaggle dataset path
... 
... def main():
...     df = pd.read_csv(DATA_PATH)
...     X = df.drop("Class", axis=1)
...     y = df["Class"]
... 
...     num_columns = X.columns
...     pre = ColumnTransformer([("scale", StandardScaler(), num_columns)])
... 
...     # imbalance handling inside pipeline
...     pipe = ImbPipeline(steps=[
...         ("pre", pre),
...         ("sm", SMOTE(random_state=42)),
...         ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
...     ])
... 
...     param_grid = {
...         "clf__C": [0.1, 1, 10],
...         "clf__penalty": ["l2"]
...     }
... 
...     gs = GridSearchCV(pipe, param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=1)
...     gs.fit(X, y)
... 
...     y_pred = gs.best_estimator_.predict(X)
...     print("ROC-AUC:", roc_auc_score(y, gs.best_estimator_.predict_proba(X)[:, 1]).round(3))
...     print(classification_report(y, y_pred))
... 
...     joblib.dump(gs.best_estimator_, "fraud_model.joblib")
...     print("Model saved → fraud_model.joblib")
... 
... if __name__ == "__main__":
...     main()
