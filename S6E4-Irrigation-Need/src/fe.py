import itertools
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import rankdata

def create_num_interactions(X, num_cols, ops=("mul", "div", "add", "sub")):
    X = X.copy()
    new_cols = []
    
    for c1, c2 in itertools.combinations(num_cols, 2):
        
        if "mul" in ops:
            col = f"{c1}_{c2}_mul"
            X[col] = X[c1] * X[c2]
            new_cols.append(col)
        
        if "div" in ops:
            col = f"{c1}_{c2}_div"
            X[col] = X[c1] / (X[c2] + 1e-6)
            new_cols.append(col)
        
        if "add" in ops:
            col = f"{c1}_{c2}_add"
            X[col] = X[c1] + X[c2]
            new_cols.append(col)
        
        if "sub" in ops:
            col = f"{c1}_{c2}_sub"
            X[col] = X[c1] - X[c2]
            new_cols.append(col)
    
    print(f"Added {len(new_cols)} numerical interaction features")
    return X, new_cols

def create_cat_bigrams(X, cat_cols):
    X = X.copy()
    new_cols = []
    
    for c1, c2 in itertools.combinations(cat_cols, 2):
        col = f"{c1}_{c2}_bigram"
        X[col] = X[c1].astype(str) + "_" + X[c2].astype(str)
        new_cols.append(col)
    
    print(f"Added {len(new_cols)} bigram features")
    return X, new_cols

def create_cat_trigrams(X, cat_cols):
    X = X.copy()
    new_cols = []
    
    for c1, c2, c3 in itertools.combinations(cat_cols, 3):
        col = f"{c1}_{c2}_{c3}_trigram"
        X[col] = (
            X[c1].astype(str) + "_" +
            X[c2].astype(str) + "_" +
            X[c3].astype(str)
        )
        new_cols.append(col)
    
    print(f"Added {len(new_cols)} trigram features")
    return X, new_cols

def create_qcut_bins(X, num_cols, q=5):
    X = X.copy()
    new_cols = []
    
    for col in num_cols:
        new_col = f"{col}_bin"
        X[new_col] = pd.qcut(
            X[col],
            q=q,
            labels=False,
            duplicates="drop"
        )
        new_cols.append(new_col)
    
    print(f"Added {len(new_cols)} qcut features")
    return X, new_cols

def create_polynomial_features(X, num_cols, degree=2):
    X = X.copy()
    
    poly = PolynomialFeatures(
        degree=degree,
        include_bias=False
    )
    
    poly_array = poly.fit_transform(X[num_cols])
    feature_names = poly.get_feature_names_out(num_cols)
    
    poly_df = pd.DataFrame(
        poly_array,
        columns=feature_names,
        index=X.index
    )
    
    new_cols = [c for c in feature_names if c not in num_cols]
    poly_df = poly_df[new_cols]
    
    X = pd.concat([X, poly_df], axis=1)
    
    print(f"Added {len(new_cols)} polynomial features")
    return X, new_cols