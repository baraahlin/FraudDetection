import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from imblearn.metrics import geometric_mean_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif

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
# RL FEATURE GROUPING (SIMPLIFIED PROXY)
# =========================
def rl_feature_grouping(X, y, num_groups=3):
    """
    Simplified RL-style grouping:
    - uses feature importance proxy (ANOVA)
    - groups features based on scores
    """
    selector = SelectKBest(score_func=f_classif, k=min(20, X.shape[1]))
    # Just fit, no need to transform the data here
    selector.fit(X, y)

    scores = selector.scores_
    scores = np.nan_to_num(scores, nan=0.0)

    indices = selector.get_support(indices=True)

    # Sort features by importance (descending)
    sorted_idx = indices[np.argsort(scores[indices])[::-1]]

    # Split into groups safely
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
    return np.array(selected).astype(int)

# =========================
# MAIN PIPELINE
# =========================
def run_pipeline(X, y, use_rl=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    results = {}

    # 1. Baseline Model (All Features)
    t0 = time.time()
    clf_base = LogisticRegression(max_iter=500, solver='liblinear', class_weight='balanced', random_state=42)
    clf_base.fit(X_train, y_train)
    y_p_base = clf_base.predict(X_test)
    t1 = time.time()
    
    # Intentionally writing X_train.shape strictly as an integer to prevent tuple errors
    features_count = int(X_train.shape[1])
    
    results["baseline"] = {
        "gmean": float(geometric_mean_score(y_test, y_p_base, average='weighted')),
        "runtime": round(t1 - t0, 3),
        "features": features_count
    }

    # 2. Simplified Feature Selection
    if use_rl:
        t2 = time.time()
        
        # Step A: Group features using ANOVA proxy
        groups = rl_feature_grouping(X_train, y_train)
        
        # Step B: Select top half from each group
        final_features = select_features_from_groups(groups)
        
        # Fallback if somehow nothing is selected
        if len(final_features) == 0: 
            final_features = np.arange(features_count)
            
        # Final evaluation
        clf = LogisticRegression(max_iter=500, solver='liblinear', class_weight='balanced', random_state=42)
        clf.fit(X_train[:, final_features], y_train)
        y_p_rl = clf.predict(X_test[:, final_features])
        t3 = time.time()
        
        results["rl_only"] = {
            "gmean": float(geometric_mean_score(y_test, y_p_rl, average='weighted')),
            "runtime": round(t3 - t2, 3),
            "features_used": int(len(final_features)),
            "total_features": features_count,
            "feature_ratio": float(len(final_features) / features_count)
        }

    return results

def format_results_table(results):
    table = []

    # Baseline
    base = results["baseline"]
    table.append([
        "Baseline (All)",
        round(base["gmean"], 4),
        base["runtime"],
        base["features"],
        1.0
    ])

    # RL
    if "rl_only" in results:
        rl = results["rl_only"]
        table.append([
            "RL-Based Selection",
            round(rl["gmean"], 4),
            
            rl["runtime"],
            rl["features_used"],
            round(rl["feature_ratio"], 2)
        ])

    df = pd.DataFrame(table, columns=[
        "Method", "G-Mean", "Runtime (s)", "Features Used", "Feature Ratio"
    ])

    return df