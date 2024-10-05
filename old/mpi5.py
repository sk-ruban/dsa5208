"""
DISTRIBUTED CONJUGATE GRADIENT
"""

import time
import numpy as np
from mpi4py import MPI
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def prepare_data():
    """
    Prepare data by:
    1. Dropping y-prediction
    2. One hot encoding OceanProximity
    3. Standardisation with Standard Scalar
    """
    data = pd.read_csv('data/housing_20k.tsv', sep='\t', header=None)
    data.columns = ['longitude', 'latitude', 'housingMedianAge', 'totalRooms',
                    'totalBedrooms', 'population', 'households', 'medianIncome',
                    'oceanProximity', 'medianHouseValue']
    
    # Convert ocean proximity to one-hot encoding
    X = pd.get_dummies(data.drop('medianHouseValue', axis=1), columns=['oceanProximity']).astype(int)
    y = data['medianHouseValue']
    
    # Split the data in training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale the data to mean = 0, sd = 1
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert the y data to float to match x post standardised
    return X_train_scaled, X_test_scaled, y_train.values.astype(np.float64), y_test.values.astype(np.float64)

def gaussian_kernel(X1, X2, sigma):
    """Compute the Gaussian kernel matrix between X1 and X2"""
    dist_matrix = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-dist_matrix / (2 * sigma**2))

def circular_kernel_computation(local_X, comm, sigma):
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

    print(f"Process {rank}: Starting kernel computation with {n_local} samples.")
    
    for i in range(size):
        # Compute the kernel between your data and the current chunk
        print(f"Process {rank}: Computing kernel chunk with data from process {(rank - i + size) % size}.")
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
        print(f"Process {rank}: Sending data to process {dest} and receiving data from process {source}.")
        comm.Sendrecv(send_data, dest=dest, recvbuf=recv_data, source=source)
        local_X = recv_data  # Update local_X for the next iteration

    print(f"Process {rank}: Completed kernel computation.")
    
    return kernel_row

def distributed_conjugate_gradient(K_local, y_local, lambda_reg, comm, max_iter=1000, tol=1e-6):
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Initializations
    x_local = np.zeros_like(y_local)
    r_local = y_local.copy()  # Since x_local is zero initially
    p_local = r_local.copy()

    # Compute initial residual norm
    r_norm_sq_local = np.dot(r_local, r_local)
    r_norm_sq_global = comm.allreduce(r_norm_sq_local, op=MPI.SUM)
    b_norm_global = np.sqrt(r_norm_sq_global)

    for iteration in range(max_iter):
        # Matrix-vector product: Ap = (K + lambda * I) * p
        # Compute K @ p in a distributed manner
        # First, gather p_local from all processes to form p_global
        p_global = np.zeros(comm.allreduce(len(p_local), op=MPI.SUM))
        counts = comm.allgather(len(p_local))
        displacements = np.insert(np.cumsum(counts), 0, 0)[0:-1]
        comm.Allgatherv(p_local, [p_global, counts, displacements, MPI.DOUBLE])

        # Compute local part of Ap
        Ap_local = K_local @ p_global  # K_local is (n_local, n_total), p_global is (n_total,)
        Ap_local += lambda_reg * p_local  # Add lambda * p_local

        # Compute scalar products
        pAp_local = np.dot(p_local, Ap_local)
        pAp_global = comm.allreduce(pAp_local, op=MPI.SUM)

        # Compute alpha
        alpha = r_norm_sq_global / pAp_global

        # Update x and r
        x_local += alpha * p_local
        r_local -= alpha * Ap_local

        # Compute new residual norm
        r_norm_sq_new_local = np.dot(r_local, r_local)
        r_norm_sq_new_global = comm.allreduce(r_norm_sq_new_local, op=MPI.SUM)

        # Check convergence
        residual = np.sqrt(r_norm_sq_new_global) / b_norm_global
        if rank == 0:
            print(f"Iteration {iteration + 1}: Relative Residual = {residual}")
        if residual < tol:
            if rank == 0:
                print("Conjugate Gradient: Converged based on relative residual.")
            break

        # Compute beta
        beta = r_norm_sq_new_global / r_norm_sq_global

        # Update p
        p_local = r_local + beta * p_local

        # Update residual norm
        r_norm_sq_global = r_norm_sq_new_global

    return x_local  # Each process returns its part of x

def kernel_ridge_regression(X_train, y_train, X_test, y_test, sigma, lambda_reg, comm):
    size = comm.Get_size()
    rank = comm.Get_rank()

    n_samples = len(X_train)
    n_features = X_train.shape[1]

    # Determine local data range
    counts = [n_samples // size + (1 if i < n_samples % size else 0) for i in range(size)]
    displacements = np.insert(np.cumsum(counts), 0, 0)[0:-1]
    start = displacements[rank]
    end = start + counts[rank]

    local_X_train = np.ascontiguousarray(X_train[start:end])
    local_y_train = np.ascontiguousarray(y_train[start:end])

    n_local = local_X_train.shape[0]
    n_total = comm.allreduce(n_local, op=MPI.SUM)

    if rank == 0:
        start_time = time.time()

    # Compute local kernel matrix rows
    K_local = circular_kernel_computation(local_X_train, comm, sigma)

    if rank == 0:
        kernel_time = time.time()
        print(f"Kernel computation time: {kernel_time - start_time} seconds")

    # Perform distributed conjugate gradient solver
    alpha_local = distributed_conjugate_gradient(K_local, local_y_train, lambda_reg, comm)

    if rank == 0:
        cg_time = time.time()
        print(f"Conjugate gradient time: {cg_time - kernel_time} seconds")
        print(f"Process {rank}: Conjugate gradient solver finished.")

    # Gather alpha_local from all processes to form alpha_global
    alpha_counts = comm.allgather(len(alpha_local))
    alpha_displacements = np.insert(np.cumsum(alpha_counts), 0, 0)[0:-1]
    alpha_global = np.zeros(comm.allreduce(len(alpha_local), op=MPI.SUM))
    comm.Allgatherv(alpha_local, [alpha_global, alpha_counts, alpha_displacements, MPI.DOUBLE])

    if rank == 0:
        print(f"Process {rank}: Computing predictions.")

    # Each process computes its part of the training predictions
    y_pred_train_local = K_local @ alpha_global
    y_pred_train = None
    if rank == 0:
        y_pred_train = np.zeros(n_samples)
    comm.Gatherv(y_pred_train_local, [y_pred_train, counts, displacements, MPI.DOUBLE], root=0)

    # Compute test kernel matrix in parallel
    n_test = X_test.shape[0]
    test_counts = [n_test // size + (1 if i < n_test % size else 0) for i in range(size)]
    test_displacements = np.insert(np.cumsum(test_counts), 0, 0)[0:-1]
    test_start = test_displacements[rank]
    test_end = test_start + test_counts[rank]
    local_X_test = X_test[test_start:test_end]

    # Each process computes its part of K_test
    K_test_local = gaussian_kernel(local_X_test, X_train, sigma)
    y_pred_test_local = K_test_local @ alpha_global

    # Gather test predictions
    y_pred_test = None
    if rank == 0:
        y_pred_test = np.zeros(n_test)
    comm.Gatherv(y_pred_test_local, [y_pred_test, test_counts, test_displacements, MPI.DOUBLE], root=0)

    if rank == 0:
        # Compute RMSE
        rmse_train = sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = sqrt(mean_squared_error(y_test, y_pred_test))
        return rmse_train, rmse_test
    else:
        return None, None

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        X_train, X_test, y_train, y_test = prepare_data()
        X_train = np.ascontiguousarray(X_train)
        X_test = np.ascontiguousarray(X_test)
        y_train = np.ascontiguousarray(y_train)
        y_test = np.ascontiguousarray(y_test)
    else:
        X_train = X_test = y_train = y_test = None

    # Broadcast data to all processes
    X_train = comm.bcast(X_train, root=0)
    X_test = comm.bcast(X_test, root=0)
    y_train = comm.bcast(y_train, root=0)
    y_test = comm.bcast(y_test, root=0)

    sigma = 1.0
    lambda_reg = 1.0

    if rank == 0:
        print(f"Using sigma: {sigma}, lambda: {lambda_reg}")

    rmse_train, rmse_test = kernel_ridge_regression(X_train, y_train, X_test, y_test, sigma, lambda_reg, comm)

    if rank == 0:
        print(f"Training RMSE: {rmse_train}")
        print(f"Test RMSE: {rmse_test}")

if __name__ == "__main__":
    main()
