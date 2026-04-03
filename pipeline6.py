import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from imblearn.metrics import geometric_mean_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# --- MFO Optimization Core ---
# --- MFO Optimization Core ---
def evaluate_fitness(X, y, selected_indices):
    if len(selected_indices) == 0: 
        return 0
    
    # Fast Logistic Regression for fitness evaluation
    model = LogisticRegression(max_iter=200, solver='liblinear', class_weight='balanced', random_state=42)
    model.fit(X[:, selected_indices], y)
    y_pred = model.predict(X[:, selected_indices])
    
    cm = confusion_matrix(y, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        return np.sqrt((tp/(tp+fn+1e-6)) * (tn/(tn+fp+1e-6)))
    return 0

def mfo_select_features(X, y, group_features, iterations=2, pop_size=5):
    n = len(group_features)
    if n == 0: return []
    
    pop = np.random.randint(0, 2, size=(pop_size, n))
    best_sol = pop 
    best_fit = -1

    for _ in range(iterations):
        for i in range(pop_size):
            idx = [group_features[j] for j in range(n) if pop[i, j] == 1]
            fit = evaluate_fitness(X, y, idx)
            if fit > best_fit:
                best_fit = fit
                best_sol = pop[i].copy()
            pop[i] = np.logical_xor(pop[i], np.random.rand(n) < 0.1).astype(int)
            
    return [group_features[i] for i in range(n) if best_sol[i] == 1]

# --- Main Pipeline Logic ---
def run_pipeline(X, y, use_rl_mfo=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    results = {}

    # 1. Baseline Model
    t0 = time.time()
    
    # Changed to Logistic Regression
    clf_base = LogisticRegression(max_iter=500, solver='liblinear', class_weight='balanced', random_state=42)
    clf_base.fit(X_train, y_train)
    y_p_base = clf_base.predict(X_test)
    t1 = time.time()
    
    results["baseline"] = {
        "gmean": float(geometric_mean_score(y_test, y_p_base)),
        "runtime": round(t1 - t0, 3),
        "features": int(X_train.shape[1]) 
    }

    # 2. RL + MFO
    if use_rl_mfo:
        from rl_module import run_rl_grouping
        t2 = time.time()
        
        assignments = run_rl_grouping(X_train, y_train)
        
        final_features = []
        for g in np.unique(assignments):
            g_idx = np.where(assignments == g)[0].flatten()
            if len(g_idx) > 0:
                final_features.extend(mfo_select_features(X_train, y_train, g_idx))
        
        final_features = np.unique(final_features).astype(int)
        
        if len(final_features) == 0: 
            final_features = np.arange(X_train.shape[1]) 
        
        # Changed to Logistic Regression
        clf = LogisticRegression(max_iter=500, solver='liblinear', class_weight='balanced', random_state=42)
        clf.fit(X_train[:, final_features], y_train)
        y_p_rl = clf.predict(X_test[:, final_features])
        t3 = time.time()
        
        results["rl_mfo"] = {
            "gmean": float(geometric_mean_score(y_test, y_p_rl)),
            "runtime": round(t3 - t2, 3),
            "features_used": int(len(final_features)),
            "total_features": int(X_train.shape[1]),
            "feature_ratio": float(len(final_features) / X_train.shape[1])
        }

    return results

def preprocess_dataset(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    if y.dtype == 'object': 
        y = y.astype('category').cat.codes
    X = pd.get_dummies(X)
    X = X.fillna(X.mean())
    return X.values, y.values