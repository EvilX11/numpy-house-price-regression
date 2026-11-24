"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
import numpy as np
def impute_nan_with_mean(X):
    x_new=X
    col_means=np.nanmean(x_new,axis=0)
    col_means=np.nan_to_num(col_means,nan=0.0)
    x_new=np.where(np.isnan(x_new),col_means,x_new)
    return x_new

# Step 2 - compute_iqr_bounds
import numpy as np
def compute_iqr_bounds(X, k=1.5):
    q1=np.percentile(X,25,axis=0)
    q3=np.percentile(X,75,axis=0)
    iqr=q3-q1
    lower=q1-k*iqr
    upper=q3+k*iqr
    return lower,upper

# Step 3 - clip_columns
import numpy as np
def clip_columns(X, lower, upper):
    x_new=np.array(X)
    n,f=x_new.shape
    for i in range(f):
        x_new[:,i]=np.where(x_new[:,i]<lower[i],lower[i],x_new[:,i])
        x_new[:,i]=np.where(x_new[:,i]>upper[i],upper[i],x_new[:,i])
    return x_new

# Step 4 - make_ratio_feature
import numpy as np
def make_ratio_feature(numerator, denominator, eps=1e-8):
    return numerator/(denominator+eps)

# Step 5 - append_column
def append_column(x, col):
    col=col.reshape(-1,1)
    x_new=np.hstack([x,col])
    return x_new

# Step 6 - one_hot_encode
import numpy as np
def one_hot_encode(labels):
    l=np.array(labels)
    f=len(np.unique(l))
    uni=np.unique(l)
    uni.sort()
    mpp={uni[i]:i for i in range(len(uni))}
    row=np.zeros((len(l),f))
    for i in range(len(labels)):
        curr_label=labels[i]
        row[i][mpp[curr_label]]=1
    return row

# Step 7 - fit_standardizer
import numpy as np
def fit_standardizer(X):
    x=np.array(X)
    m=np.mean(x,axis=0)
    s=np.std(x,axis=0)
    s=np.where(s==0,1.0,s)
    return m,s

# Step 8 - apply_standardizer
import numpy as np
def apply_standardizer(X, mean, std):
    x_new=np.array(X)
    m=mean.reshape((1,-1))
    s=std.reshape((1,-1))
    return (x_new-m)/s

# Step 9 - add_bias_column
import numpy as np
def add_bias_column(X):
    x=np.array(X)
    m,n=x.shape
    o=np.ones((m,1))
    x=np.hstack([o,x])
    return x

# Step 10 - make_shuffled_indices
import numpy as np
def make_shuffled_indices(n_samples, seed):
    rnd=np.random.default_rng(seed=seed)
    return rnd.permutation(n_samples)

# Step 11 - partition_indices
import math
def partition_indices(indices, train_ratio, val_ratio):
    start=0
    l=len(indices)
    end=math.floor(train_ratio*l)
    train_idx=indices[start:start+end]
    start=start+end
    end=math.floor(val_ratio*l)
    val_idx=indices[start:start+end]
    start=start+end
    test_idx=indices[start:]
    return (train_idx,val_idx,test_idx)

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    return (X[indices],y[indices])

# Step 13 - ols_fit
import numpy as np
def ols_fit(X, y):
    x=np.array(X)
    y_new=np.array(y)
    #normal equation" (X.T@X)theta=X.T@y
    A=x.T@x
    b=x.T@y_new
    return np.linalg.solve(A,b)

# Step 14 - ols_predict
import numpy as np
def ols_predict(X, theta):
    x=np.array(X)
    #y_hat=sum(w_i*theta_i)+bias
    return X@theta

# Step 15 - mean_absolute_error
import numpy as np
def mean_absolute_error(y_true, y_pred):
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)
    diff=y_true-y_pred
    diff=np.abs(diff)
    return np.sum(diff)/len(y_true)

# Step 16 - root_mean_squared_error
import numpy as np
def root_mean_squared_error(y_true, y_pred):
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)
    diff=(y_true-y_pred)**2
    return (np.sum(diff)/len(y_true))**.5

# Step 17 - r_squared
import numpy as np
def r_squared(y_true, y_pred):
    mean=np.sum(y_true)/len(y_true)
    ss_res=np.sum((y_true-y_pred)**2)
    ss_tot=np.sum((y_true-mean)**2)
    if ss_tot==0:
        return 0.0
    return 1-ss_res/ss_tot

# Step 18 - residual_summary
import numpy as np
def residual_summary(y_true, y_pred):
    r=y_true-y_pred
    r_mean=np.sum(r)/len(r)
    std=np.std(r)
    dict={
        'mean':r_mean,
        'std':std,
        'median_abs':np.median(np.abs(r))
    }
    return dict

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    x=impute_nan_with_mean(X)
    lower,upper=compute_iqr_bounds(x,k=iqr_k)
    x=clip_columns(x,lower,upper)
    return x

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

