# with mfo
#RL (grouping) → MFO (optimize features inside groups) → classifier

import numpy as np
import pandas as pd
import time as time
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

    start_time = time.time()  # ⏱️ start

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    original_feature_count = X.shape[1]

    if use_rl:
        from rl_module import run_rl_grouping

        assignments = run_rl_grouping(X_train, y_train)

        selected_features = []

        for g in np.unique(assignments):
            idx = np.where(assignments == g)[0]

            if len(idx) == 0:
                continue

            # 🔥 MFO optimization inside group
            best_subset = mfo_select_features(X_train, y_train, idx)

            selected_features.extend(best_subset)
        
        # ✅ safety fallback (AFTER loop)
        if len(selected_features) == 0:
            selected_features = np.arange(X_train.shape[1])

        #selected_features = np.array(selected_features)
        selected_features = np.unique(selected_features)

        X_train = X_train[:, selected_features]
        X_test = X_test[:, selected_features]

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('logistic', LogisticRegression(max_iter=200, solver='liblinear'))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    end_time = time.time()  # ⏱️ end

    results = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "gmean": float(geometric_mean_score(y_test, y_pred)),
        "features_used": int(X_train.shape[1]),
        "total_features": int(original_feature_count),
        "feature_ratio": round(X_train.shape[1] / original_feature_count, 2),
        "runtime_seconds": round(end_time - start_time, 2)  # 🔥 NEW
    }

    return results
'''
without time 
def run_pipeline(X, y, use_rl=True):
    

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print("Train class distribution:", np.unique(y_train, return_counts=True))
    print("Test class distribution:", np.unique(y_test, return_counts=True))

    if len(np.unique(y)) < 2:
        return {"error": "Only one class detected in dataset"}
    
    if use_rl:
        from rl_module import run_rl_grouping

        assignments = run_rl_grouping(X_train, y_train)

        selected_features = []

        for g in np.unique(assignments):
            idx = np.where(assignments == g)[0]
            if len(idx) > 0:
                selected_features.extend(idx[:max(1, len(idx)//2)])

        selected_features = np.array(selected_features)

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
'''

#mfo optimizer 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def evaluate_subset(X, y, features):
    if len(features) == 0:
        return 0

    X_sub = X[:, features]

    model = LogisticRegression(max_iter=200, solver='liblinear')
    model.fit(X_sub, y)
    y_pred = model.predict(X_sub)

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    sensitivity = tp / (tp + fn + 1e-6)
    specificity = tn / (tn + fp + 1e-6)

    gmean = np.sqrt(sensitivity * specificity)

    return gmean


def mfo_select_features(X, y, group_features, iterations=5, population_size=5):

    n = len(group_features)
    if n == 0:
        return []

    # initialize population
    population = [
        np.random.randint(0, 2, size=n)
        for _ in range(population_size)
    ]

    best_solution = None
    best_score = -1

    for _ in range(iterations):
        for individual in population:

            selected = [group_features[i] for i in range(n) if individual[i] == 1]

            score = evaluate_subset(X, y, selected)

            if score > best_score:
                best_score = score
                best_solution = individual.copy()

        # update population (simple mutation)
        for i in range(population_size):
            mutation = np.random.rand(n) < 0.1
            population[i] = np.logical_xor(population[i], mutation).astype(int)

    # final selected features
    final_features = [
        group_features[i]
        for i in range(n)
        if best_solution[i] == 1
    ]

    return final_features