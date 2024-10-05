import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Data Preparation
def prepare_data():
    # Import Data
    df = pd.read_csv('data/housing.tsv', sep='\t', header=None)
    
    df.columns = ['longitude', 'latitude', 'housingMedianAge', 'totalRooms',
                  'totalBedrooms', 'population', 'households', 'medianIncome',
                  'oceanProximity', 'medianHouseValue']

    features = ['longitude', 'latitude', 'housingMedianAge', 'totalRooms',
                'totalBedrooms', 'population', 'households', 'medianIncome',
                'oceanProximity']

    # Normalize Data
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # Split Data
    X = df[features].values
    y = df['medianHouseValue'].values.astype(float)  # Convert to float
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test

# Kernel Computation
def compute_gaussian_kernel(X1, X2, gamma):
    dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * dists)

# Conjugate Gradient Method
def conjugate_gradient(A, b, max_iter=1000, tol=1e-6):
    x = np.zeros_like(b, dtype=float)  # Ensure float dtype
    r = b - A @ x
    p = r.copy()
    r_norm_sq = np.dot(r, r)
    
    for _ in range(max_iter):
        Ap = A @ p
        alpha = r_norm_sq / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        r_norm_sq_new = np.dot(r, r)
        if np.sqrt(r_norm_sq_new) < tol:
            break
        beta = r_norm_sq_new / r_norm_sq
        p = r + beta * p
        r_norm_sq = r_norm_sq_new
    
    return x

# Prediction function
def predict(X_new, X_train, alpha, gamma):
    K_new = compute_gaussian_kernel(X_new, X_train, gamma)
    return K_new @ alpha

def main():
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data()

    # Split data for MPI
    chunk_size = len(X_train) // size
    start = rank * chunk_size
    end = start + chunk_size if rank < size - 1 else len(X_train)
    X_chunk = X_train[start:end]

    # Kernel Ridge Regression
    gamma = 1.0  # TUNE PARAMETER
    lambda_ = 0.1  # TUNE PARAMETER

    # Compute local kernel matrix
    local_K = compute_gaussian_kernel(X_chunk, X_train, gamma)
    
    # Gather all local kernel matrices
    gathered_K = comm.gather(local_K, root=0)

    if rank == 0:
        # Combine gathered kernel matrices
        K = np.vstack(gathered_K)
        
        # Solve for alpha
        A = K + lambda_ * np.eye(K.shape[0])
        alpha = conjugate_gradient(A, y_train)

        # Predictions and evaluation
        y_train_pred = predict(X_train, X_train, alpha, gamma)
        y_test_pred = predict(X_test, X_train, alpha, gamma)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        print(f"Train RMSE: ${train_rmse:.2f}")
        print(f"Test RMSE: ${test_rmse:.2f}")

if __name__ == "__main__":
    main()