import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from imblearn.metrics import geometric_mean_score

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression

# =========================
# PREPROCESS
# =========================
def preprocess_dataset(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    if y.dtype == 'object':
        y = y.astype('category').cat.codes

    X = pd.get_dummies(X)
    X = X.fillna(X.mean())

    return X.values, y.values


# =========================
# RL FEATURE GROUPING (SIMPLIFIED)
# =========================
def rl_feature_grouping(X, y, num_groups=3):
    """
    Simplified RL-style grouping:
    - uses feature importance proxy (ANOVA)
    - groups features based on scores
    """

    selector = SelectKBest(score_func=f_classif, k=min(20, X.shape[1]))
    X_new = selector.fit_transform(X, y)

    scores = selector.scores_
    indices = selector.get_support(indices=True)

    # Sort features by importance
    sorted_idx = indices[np.argsort(scores[indices])]

    # Split into groups
    groups = np.array_split(sorted_idx, num_groups)

    return groups


# =========================
# FEATURE SELECTION (LIGHT MFO STYLE)
# =========================
def select_features_from_groups(groups):
    selected = []

    for g in groups:
        if len(g) > 0:
            selected.extend(g[:max(1, len(g)//2)])  # take top half

    return np.array(selected)


# =========================
# MAIN PIPELINE
# =========================
def run_pipeline(X, y, use_rl=True):
    

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print("Train class distribution:", np.unique(y_train, return_counts=True))
    print("Test class distribution:", np.unique(y_test, return_counts=True))

    if len(np.unique(y)) < 2:
        return {"error": "Only one class detected in dataset"}
    
    if use_rl:
        groups = rl_feature_grouping(X_train, y_train)
        selected_features = select_features_from_groups(groups)

        X_train = X_train[:, selected_features]
        X_test = X_test[:, selected_features]

    model = Pipeline([
        ('scaler', StandardScaler()),
        #('smote', SMOTE()),
        ('logistic', LogisticRegression(max_iter=200,class_weight='balanced'))
        #('knn', KNeighborsClassifier(n_neighbors=5))
    ])



    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "gmean": float(geometric_mean_score(y_test, y_pred)),
        "features_used": int(X_train.shape[1])
    }

    return results