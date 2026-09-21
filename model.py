"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    
    means = np.nanmean(X,axis=0)

    means[np.isnan(means)] = 0.0

    for col in range(X.shape[1]):
        X[np.isnan(X[:,col]),col] = means[col]

    return X

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):

    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)

    iqr = q3 - q1

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    return (lower,upper)

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    
    cliped = np.clip(X,lower,upper)

    return cliped

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    
    res = numerator / (denominator + eps)

    return res

# Step 5 - append_column
def append_column(X, col):
    
    n:int = X.shape[0]

    if len(col) != n:
        raise ValueError(f"Wrong column shape, which should be {(n,1)}")

    col = col.reshape(-1,1)

    X = np.concatenate([X,col],axis=1)

    return X

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    
    uniqe = np.unique(labels)
    uniqe.sort()

    N:int = len(labels)
    C:int = len(uniqe)

    encoded = np.zeros((N,C))

    mask = labels[:,None] == uniqe[None,:]
    encoded[mask] = 1.0

    return encoded

# Step 7 - fit_standardizer
def fit_standardizer(X):
    
    mean = np.mean(X,axis=0)
    std = np.std(X,axis=0)

    std[std==0.0] = 1.0

    return (mean,std)

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    
    return (X - mean) / std

# Step 9 - add_bias_column
def add_bias_column(X):
    N:int = X.shape[0]

    bias = np.ones((N,1))

    res = np.concatenate([bias,X], axis=1)

    return res

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    np.random.seed(seed)

    return np.random.permutation(n_samples)

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    N:int = len(indices)

    train_split = int(np.floor(N * train_ratio))
    val_split = int(np.floor(N * val_ratio)) + train_split

    train_idx = indices[:train_split]
    val_idx = indices[train_split:val_split]
    test_idx = indices[val_split:]

    return (train_idx,val_idx,test_idx)

# Step 12 - subset_xy (not yet solved)
# TODO: implement

# Step 13 - ols_fit (not yet solved)
# TODO: implement

# Step 14 - ols_predict (not yet solved)
# TODO: implement

# Step 15 - mean_absolute_error (not yet solved)
# TODO: implement

# Step 16 - root_mean_squared_error (not yet solved)
# TODO: implement

# Step 17 - r_squared (not yet solved)
# TODO: implement

# Step 18 - residual_summary (not yet solved)
# TODO: implement

# Step 19 - prepare_cleaned_features (not yet solved)
# TODO: implement

# Step 20 - assemble_feature_matrix (not yet solved)
# TODO: implement

# Step 21 - make_train_val_test (not yet solved)
# TODO: implement

# Step 22 - standardize_and_add_bias (not yet solved)
# TODO: implement

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

