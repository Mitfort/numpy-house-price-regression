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

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    return X[indices], y[indices]

# Step 13 - ols_fit
def ols_fit(X, y):
    theta = np.linalg.inv(X.T @ X) @ X.T @ y

    return theta

# Step 14 - ols_predict
def ols_predict(X, theta):
    pred = X @ theta 

    return pred

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    
    return np.mean(np.abs(y_true-y_pred))

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    
    return np.sqrt(np.mean((y_true - y_pred)**2) )

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    mean = np.mean(y_true)

    ssr = np.sum((y_true - y_pred)**2)
    sst = np.sum((y_true - mean)**2)

    if sst == 0.0: return 0.0

    return 1 - (ssr/sst)

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    r = y_true - y_pred

    mean = np.mean(r)
    std = np.std(r)

    median_abs = np.median(np.abs(r))

    return {
        "mean": mean,
        "std": std,
        "median_abs": median_abs
    }

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    
    X_filled = impute_nan_with_mean(X)
    lower,upper = compute_iqr_bounds(X_filled,iqr_k)

    X_clipped = clip_columns(X_filled,lower,upper)

    return X_clipped

# Step 20 - assemble_feature_matrix
import numpy as np
def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    
    num = X_num[:,ratio_num_idx]
    den = X_num[:,ratio_den_idx]

    ratio_column = make_ratio_feature(num,den)

    matrix = append_column(X_num,ratio_column)

    if cat_labels is not None:
        one_hot = one_hot_encode(cat_labels)
        matrix = np.concatenate([matrix,one_hot],axis=1)

    return matrix

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    N:int = X.shape[0]

    idxs = make_shuffled_indices(N, seed)

    train_idx, val_idx, test_idx = partition_indices(idxs,train_ratio,val_ratio)

    X_train,y_train = subset_xy(X,y,train_idx)
    X_val,y_val = subset_xy(X,y,val_idx)
    X_test,y_test = subset_xy(X,y,test_idx)

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }

# Step 22 - standardize_and_add_bias
def standardize_and_add_bias(splits):
    mean,std = fit_standardizer(splits['X_train'])

    X_train_scaled = apply_standardizer(splits['X_train'], mean, std)
    X_val_scaled = apply_standardizer(splits['X_val'], mean, std)
    X_test_scaled = apply_standardizer(splits['X_test'], mean, std)

    X_train_scaled = add_bias_column(X_train_scaled)
    X_val_scaled = add_bias_column(X_val_scaled)
    X_test_scaled = add_bias_column(X_test_scaled)

    splits_scaled = {
        'X_train':  X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train': splits['y_train'],
        'y_val': splits['y_val'],
        'y_test': splits['y_test']
    }

    return (splits_scaled,mean,std)

# Step 23 - evaluate_predictions
def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true,y_pred)
    rmse = root_mean_squared_error(y_true,y_pred)
    r_2 = r_squared(y_true,y_pred)
    res_summary = residual_summary(y_true,y_pred)

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r_2,
        'residual_summary': res_summary
    }

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

