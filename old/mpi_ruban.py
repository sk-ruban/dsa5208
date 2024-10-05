"""
Train Root Mean Squared Error: 50304.6023141701
Test Root Mean Squared Error: 58715.42076386794
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from mpi4py import MPI

def prepare_data():
    df = pd.read_csv('data/housing.tsv', sep='\t', header=None)

    df.columns = ['longitude', 'latitude', 'housingMedianAge', 'totalRooms',
                  'totalBedrooms', 'population', 'households', 'medianIncome',
                  'oceanProximity', 'medianHouseValue']
    
    numeric_features = ['longitude', 'latitude', 'housingMedianAge', 'totalRooms',
                        'totalBedrooms', 'population', 'households', 'medianIncome']
    categorical_features = ['oceanProximity']

    # Create preprocessing steps
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Fit the preprocessor and transform the features
    X = preprocessor.fit_transform(df.drop('medianHouseValue', axis=1))
    y = df['medianHouseValue'].values

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # FOR TESTING - CAN DELETE LATER
    numeric_feature_names = numeric_features
    categorical_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = numeric_feature_names + list(categorical_feature_names)
    X_df = pd.DataFrame(X, columns=feature_names)

    return X_train, X_test, y_train, y_test, X_df

def gaussian_kernel(X1, X2, sigma):
    """Compute the Gaussian kernel matrix between X1 and X2."""
    dist_matrix = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-dist_matrix / (2 * sigma**2))

def conjugate_gradient(A, b, max_iter=1000, tol=1e-6):
    """Compute the Gaussian kernel matrix between X1 and X2."""
    x = np.zeros_like(b, dtype=float)  # Ensure float dtype
    r = b - A @ x
    p = r.copy()
    r_norm_sq = np.dot(r, r)
    
    for iteration in range(max_iter):
        Ap = A @ p
        alpha = r_norm_sq / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        r_norm_sq_new = np.dot(r, r)
        residual = np.sqrt(r_norm_sq_new)
        print(f"Iteration {iteration + 1}: Residual = {residual}")
        if residual < tol:
            break
        beta = r_norm_sq_new / r_norm_sq
        p = r + beta * p
        r_norm_sq = r_norm_sq_new
    
    return x

def kernel_ridge_regression(X_train, y_train, X_test, sigma, lambda_reg):
    """Perform kernel ridge regression."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # Distribute the computation of the kernel matrix
    local_n = n_train // size + (1 if rank < n_train % size else 0)
    start = rank * (n_train // size) + min(rank, n_train % size)
    end = start + local_n

    local_K = gaussian_kernel(X_train[start:end], X_train, sigma)
    K = np.empty((n_train, n_train), dtype=np.float64)
    comm.Allgatherv(local_K, (K, [n * n_train for n in comm.allgather(local_n)]))

    # Solve (K + lambda * I) * alpha = y
    if rank == 0:
        alpha = conjugate_gradient(K + lambda_reg * np.eye(n_train), y_train)
    else:
        alpha = None
    alpha = comm.bcast(alpha, root=0)

    # Compute predictions for train set
    local_train_predictions = np.dot(local_K, alpha)
    train_predictions = np.empty(n_train, dtype=np.float64)
    comm.Allgatherv(local_train_predictions, (train_predictions, [n for n in comm.allgather(local_n)]))

    # Compute predictions for test set
    local_K_test = gaussian_kernel(X_test[start:end], X_train, sigma)
    local_test_predictions = np.dot(local_K_test, alpha)
    test_predictions = np.empty(n_test, dtype=np.float64)
    comm.Allgatherv(local_test_predictions, (test_predictions, [n for n in comm.allgather(local_n)]))

    return train_predictions, test_predictions

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    X_train, X_test, y_train, y_test, X_df = prepare_data()

    # FOR TESTING - CAN DELETE LATER
    """
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of features: {X_train.shape[1]}")
    print("\nFirst few rows of the processed data:")
    print(X_df.head())
    print("\nFeature names:")
    print(X_df.columns)
    """

    # Hyperparameters (you may want to tune these)
    sigma = 1.0
    lambda_reg = 1.0

    train_predictions, test_predictions = kernel_ridge_regression(X_train, y_train, X_test, sigma, lambda_reg)

    if rank == 0:
        train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        print(f"Train Root Mean Squared Error: {train_rmse}")
        print(f"Test Root Mean Squared Error: {test_rmse}")

if __name__ == "__main__":
    main()