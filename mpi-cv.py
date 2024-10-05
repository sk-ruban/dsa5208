import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from mpi4py import MPI
from sklearn.metrics import mean_squared_error
from math import sqrt

def prepare_data():
    """
    Prepare data by:
    1. Loading and preprocessing the data
    2. Performing feature engineering
    3. Handling infinite and NaN values
    4. Splitting the data into train and test sets
    5. Standardizing the features (not the target variable)
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

    data = data.drop(['totalRooms', 'totalBedrooms'], axis=1)

    # Convert ocean proximity to one-hot encoding
    X = pd.get_dummies(data.drop('medianHouseValue', axis=1), columns=['oceanProximity'])
    y = data['medianHouseValue'].values.astype(np.float64)  # Convert to NumPy array
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.3, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return (X_train_scaled, X_test_scaled, y_train, y_test)

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
        if residual < tol:
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
    
    # Keep a copy of original data
    original_local_X = local_X.copy()
    kernel_row = np.zeros((n_local, n_total))

    for i in range(size):
        # Compute the kernel between data and the current chunk
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
        comm.Sendrecv(send_data, dest=dest, recvbuf=recv_data, source=source)
        local_X = recv_data 
    
    return kernel_row

def kernel_ridge_regression(X_train, y_train, X_test, sigma, lambda_reg, comm):
    size = comm.Get_size()
    rank = comm.Get_rank()

    n_samples = len(X_train)
    # Compute start and end indices for each process
    indices = np.linspace(0, n_samples, num=size+1, dtype=int)
    start = indices[rank]
    end = indices[rank+1]
    local_X_train = np.ascontiguousarray(X_train[start:end])
    local_y_train = np.ascontiguousarray(y_train[start:end])

    if local_X_train.shape[0] == 0:
        local_K = np.empty((0, n_samples))
    else:
        local_K = circular_kernel_computation(local_X_train, comm, sigma)

    K = comm.gather(local_K, root=0)
    y_gathered = comm.gather(local_y_train, root=0)

    if rank == 0:
        K = np.vstack(K)
        y_train_full = np.concatenate(y_gathered)

        n = K.shape[0]
        A = K + lambda_reg * np.eye(n)
        alpha = conjugate_gradient(A, y_train_full)

        K_test = gaussian_kernel(X_test, X_train, sigma)
        y_pred = K_test @ alpha

        return y_pred
    else:
        return None

def cross_validate_mpi_krr(X, y, param_grid, n_splits=5):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    best_score = float('inf') 
    best_params = None

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for sigma in param_grid['sigma']:
        for lambda_reg in param_grid['lambda_reg']:
            scores = []
            if rank == 0:
                print(f"Params: sigma={sigma}, lambda_reg={lambda_reg}")
            for train_index, val_index in kf.split(X):
                if rank == 0:
                    X_train_cv, X_val_cv = X[train_index], X[val_index]
                    y_train_cv, y_val_cv = y[train_index], y[val_index]
                else:
                    X_train_cv = X_val_cv = y_train_cv = y_val_cv = None

                # Broadcast the training and validation data to all processes
                X_train_cv = comm.bcast(X_train_cv, root=0)
                y_train_cv = comm.bcast(y_train_cv, root=0)
                X_val_cv = comm.bcast(X_val_cv, root=0)
                y_val_cv = comm.bcast(y_val_cv, root=0)

                y_pred_cv = kernel_ridge_regression(X_train_cv, y_train_cv, X_val_cv, sigma, lambda_reg, comm)

                if rank == 0:
                    rmse = sqrt(mean_squared_error(y_val_cv, y_pred_cv))
                    scores.append(rmse)
            if rank == 0:
                avg_score = np.mean(scores)
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = {'sigma': sigma, 'lambda_reg': lambda_reg}
                print(f"RMSE: {avg_score}")

    if rank == 0:
        return best_params, best_score
    else:
        return None, None

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        X_train, X_test, y_train, y_test = prepare_data()
    else:
        X_train = X_test = y_train = y_test = None

    # Broadcast data to all processes
    X_train = comm.bcast(X_train, root=0)
    X_test = comm.bcast(X_test, root=0)
    y_train = comm.bcast(y_train, root=0)
    y_test = comm.bcast(y_test, root=0)

    # Define parameter grid
    param_grid = {
        'sigma': [2.0, 2.3, 2.5],
        'lambda_reg': [0.01, 0.03, 0.05]
    }

    # Perform cross-validation
    best_params, best_score = cross_validate_mpi_krr(X_train, y_train, param_grid)

    if rank == 0:
        print("Best parameters:", best_params)
        print("Best RMSE:", best_score)

        # Use the best parameters to train on full training set and predict on test set
        y_pred_train = kernel_ridge_regression(X_train, y_train, X_train, best_params['sigma'], best_params['lambda_reg'], comm)
        y_pred_test = kernel_ridge_regression(X_train, y_train, X_test, best_params['sigma'], best_params['lambda_reg'], comm)

        # Apply clipping
        y_pred_train = np.clip(y_pred_train, 0, 500001)
        y_pred_test = np.clip(y_pred_test, 0, 500001)

        rmse_train = sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = sqrt(mean_squared_error(y_test, y_pred_test))

        print(f"Final Training RMSE: {rmse_train}")
        print(f"Final Test RMSE: {rmse_test}")

if __name__ == "__main__":
    main()
