"""
FEATURE SELECTION + Y SCALAR
Training RMSE: 50888.32629359395
Test RMSE: 54117.04776023324
"""
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mpi4py import MPI
from sklearn.metrics import mean_squared_error
from math import sqrt

def prepare_data():
    """
    Prepare data by:
    1. Loading and preprocessing the data
    2. Performing feature engineering
    3. Selecting the most important features
    4. Splitting the data into train and test sets
    5. Standardizing the features
    """
    # Load data
    data = pd.read_csv('data/housing_20k.tsv', sep='\t', header=None)
    data.columns = ['longitude', 'latitude', 'housingMedianAge', 'totalRooms',
                    'totalBedrooms', 'population', 'households', 'medianIncome',
                    'oceanProximity', 'medianHouseValue']
    
    # Feature engineering
    data['longitude_latitude_interaction'] = data['longitude'] * data['latitude']
    data['rooms_per_household'] = data['totalRooms'] / data['households']
    data['bedrooms_per_room'] = data['totalBedrooms'] / data['totalRooms']
    data['population_per_household'] = data['population'] / data['households']
    
    # Log transform some features
    data['medianIncome'] = np.log1p(data['medianIncome'])
    data['housingMedianAge'] = np.log1p(data['housingMedianAge'])

    data = data.drop('totalRooms', axis=1)
    data = data.drop('totalBedrooms', axis=1)

    # Convert ocean proximity to one-hot encoding
    X = pd.get_dummies(data.drop('medianHouseValue', axis=1), columns=['oceanProximity'])
    y = data['medianHouseValue']
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return (X_train_scaled, X_test_scaled, 
            y_train.astype(np.float64), 
            y_test.astype(np.float64),
            scaler)

def gaussian_kernel(X1, X2, sigma):
    """Compute the Gaussian kernel matrix between X1 and X2"""
    dist_matrix = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-dist_matrix / (2 * sigma**2))

def conjugate_gradient(A, b, max_iter=1000, tol=1e-6):
    """Solve the linear system Ax = b using the Conjugate Gradient method"""
    x = np.zeros_like(b, dtype=float) 
    r = b - A @ x
    p = r.copy()
    r_norm_sq = np.dot(r, r)
    b_norm = np.linalg.norm(b)

    for iteration in range(max_iter):
        Ap = A @ p
        alpha = r_norm_sq / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        r_norm_sq_new = np.dot(r, r)
        residual = np.sqrt(r_norm_sq_new) / b_norm 
        # print(f"Iteration {iteration + 1}: Relative Residual = {residual}")
        if residual < tol:
            # print("Conjugate Gradient: Converged based on relative residual.")
            break
        beta = r_norm_sq_new / r_norm_sq
        p = r + beta * p
        r_norm_sq = r_norm_sq_new

    return x

def circular_kernel_computation(local_X, comm, sigma):
    """Compute the Gaussian kernel matrix using a circular communication pattern in MPI."""
    size = comm.Get_size()
    rank = comm.Get_rank()
    n_local = local_X.shape[0]
    
    # Gather n_local from all processes
    n_locals = comm.allgather(n_local)
    displacements = [sum(n_locals[:i]) for i in range(size)]
    n_total = sum(n_locals)
    
    # Keep a copy of your original data
    original_local_X = local_X.copy()
    kernel_row = np.zeros((n_local, n_total))

    # print(f"Process {rank}: Starting kernel computation with {n_local} samples.")
    
    for i in range(size):
        # Compute the kernel between your data and the current chunk
        # print(f"Process {rank}: Computing kernel chunk with data from process {(rank - i + size) % size}.")
        kernel_chunk = gaussian_kernel(original_local_X, local_X, sigma)
        
        # Determine the source rank of the current chunk
        source_rank = (rank - i + size) % size
        start_col = displacements[source_rank]
        end_col = start_col + n_locals[source_rank]
        kernel_row[:, start_col:end_col] = kernel_chunk
        
        # Prepare for the next iteration
        send_data = local_X.copy()
        recv_n_local = n_locals[(rank - 1 + size) % size]
        recv_data = np.empty((recv_n_local, local_X.shape[1]), dtype=local_X.dtype)
        
        dest = (rank + 1) % size
        source = (rank - 1 + size) % size
        # print(f"Process {rank}: Sending data to process {dest} and receiving data from process {source}.")
        comm.Sendrecv(send_data, dest=dest, recvbuf=recv_data, source=source)
        local_X = recv_data  # Update local_X for the next iteration

    # print(f"Process {rank}: Completed kernel computation.")
    
    return kernel_row

def kernel_ridge_regression(X_train, y_train, X_test, y_test, sigma, lambda_reg, comm, target_scaler):
    size = comm.Get_size()
    rank = comm.Get_rank()

    n_samples = len(X_train)
    local_n = n_samples // size
    start = rank * local_n
    end = start + local_n if rank < size - 1 else n_samples
    local_X_train = np.ascontiguousarray(X_train[start:end])
    local_y_train = np.ascontiguousarray(y_train[start:end])
    
    if rank == 0:
        start_time = time.time()

    local_K = circular_kernel_computation(local_X_train, comm, sigma)

    # print(f"Process {rank}: Gathering local kernel matrices and labels.")
    K = comm.gather(local_K, root=0)
    y_gathered = comm.gather(local_y_train, root=0)

    if rank == 0:
        # print(f"Process {rank}: Assembling full kernel matrix and labels.")
        K = np.vstack(K)
        y_train_full = np.concatenate(y_gathered)

        kernel_time = time.time()
        # print(f"Kernel computation time: {kernel_time - start_time} seconds")

        n = K.shape[0]
        # print(f"Process {rank}: Starting conjugate gradient solver.")
        A = K + lambda_reg * np.eye(n)
        alpha = conjugate_gradient(A, y_train_full)
        # print(f"Process {rank}: Conjugate gradient solver finished.")

        cg_time = time.time()
        # print(f"Conjugate gradient time: {cg_time - kernel_time} seconds")
    
        # print(f"Process {rank}: Computing predictions.")    

        K_test = gaussian_kernel(X_test, X_train, sigma)
        y_pred_train = K @ alpha
        y_pred_test = K_test @ alpha

        # Apply clipping after inverse transform
        y_pred_train = np.clip(y_pred_train, 0, 500001)
        y_pred_test = np.clip(y_pred_test, 0, 500001)

        rmse_train = sqrt(mean_squared_error(y_train_full, y_pred_train))
        rmse_test = sqrt(mean_squared_error(y_test, y_pred_test))

        return rmse_train, rmse_test
    else:
        return None, None
    
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        X_train, X_test, y_train, y_test, target_scaler = prepare_data()
        X_train = np.ascontiguousarray(X_train)
        X_test = np.ascontiguousarray(X_test)
        y_train = np.ascontiguousarray(y_train)
        y_test = np.ascontiguousarray(y_test)
    else:
        X_train = X_test = y_train = y_test = target_scaler = None

    # Broadcast data to all processes
    X_train = comm.bcast(X_train, root=0)
    X_test = comm.bcast(X_test, root=0)
    y_train = comm.bcast(y_train, root=0)
    y_test = comm.bcast(y_test, root=0)
    target_scaler = comm.bcast(target_scaler, root=0)

    sigma = 2.0
    lambda_reg = 0.1

    if rank == 0:
        print(f"Using sigma: {sigma}, lambda: {lambda_reg}")

    rmse_train, rmse_test = kernel_ridge_regression(X_train, y_train, X_test, y_test, sigma, lambda_reg, comm, target_scaler)

    if rank == 0:
        print(f"Training RMSE: {rmse_train}")
        print(f"Test RMSE: {rmse_test}")

if __name__ == "__main__":
    main()