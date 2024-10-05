import numpy as np
from mpi4py import MPI
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def prepare_data():
    data = pd.read_csv('data/housing.tsv', sep='\t', header=None)
    data.columns = ['longitude', 'latitude', 'housingMedianAge', 'totalRooms',
                    'totalBedrooms', 'population', 'households', 'medianIncome',
                    'oceanProximity', 'medianHouseValue']
    
    X = pd.get_dummies(data.drop('medianHouseValue', axis=1), columns=['oceanProximity'])
    y = data['medianHouseValue']

    print(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train.values.astype(np.float64), y_test.values.astype(np.float64)

def gaussian_kernel(X1, X2, gamma):
    """Compute the Gaussian kernel matrix between X1 and X2."""
    dist_matrix = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * dist_matrix)

def conjugate_gradient(A, b, max_iter=1000, tol=1e-6):
    """Solve Ax = b using the conjugate gradient method."""
    x = np.zeros_like(b, dtype=np.float64)
    r = b - A @ x
    p = r.copy()
    r_norm_sq = np.dot(r, r)

    for iteration in range(max_iter):
        Ap = A @ p
        alpha = r_norm_sq / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        r_norm_sq_new = np.dot(r, r)
        residual = sqrt(r_norm_sq_new)
        if residual < tol:
            break
        beta = r_norm_sq_new / r_norm_sq
        p = r + beta * p
        r_norm_sq = r_norm_sq_new

    return x

def kernel_ridge_regression(X_train, y_train, X_test, gamma, alpha):
    """Perform kernel ridge regression."""
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    K = gaussian_kernel(X_train, X_train, gamma)
    beta = conjugate_gradient(K + alpha * np.eye(n_train), y_train)

    train_predictions = np.dot(K, beta)
    K_test = gaussian_kernel(X_test, X_train, gamma)
    test_predictions = np.dot(K_test, beta)

    return train_predictions, test_predictions

def ensemble_kernel_ridge_regression(X_train, y_train, X_test, gamma, alpha, n_models):
    """Perform ensemble kernel ridge regression using MPI."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Determine the number of models each process will train
    models_per_process = n_models // size
    extra_models = n_models % size
    
    if rank < extra_models:
        local_n_models = models_per_process + 1
    else:
        local_n_models = models_per_process

    # Train local models
    local_train_predictions = []
    local_test_predictions = []

    for _ in range(local_n_models):
        # Create a random subset of the training data
        subset_indices = np.random.choice(X_train.shape[0], size=X_train.shape[0] // 2, replace=False)
        X_subset = X_train[subset_indices]
        y_subset = y_train[subset_indices]

        train_pred, test_pred = kernel_ridge_regression(X_subset, y_subset, X_test, gamma, alpha)
        local_train_predictions.append(train_pred)
        local_test_predictions.append(test_pred)

    # Gather all predictions
    all_train_predictions = comm.gather(local_train_predictions, root=0)
    all_test_predictions = comm.gather(local_test_predictions, root=0)

    if rank == 0:
        # Flatten the list of predictions
        all_train_predictions = [pred for sublist in all_train_predictions for pred in sublist]
        all_test_predictions = [pred for sublist in all_test_predictions for pred in sublist]

        # Average the predictions
        ensemble_train_predictions = np.mean(all_train_predictions, axis=0)
        ensemble_test_predictions = np.mean(all_test_predictions, axis=0)

        return ensemble_train_predictions, ensemble_test_predictions
    else:
        return None, None

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    X_train, X_test, y_train, y_test = prepare_data()

    # Hyperparameters
    gamma = 1.0 / (X_train.shape[1] * np.var(X_train))
    alpha = 1.0
    n_models = 10  # Number of models in the ensemble

    if rank == 0:
        print(f"Using gamma: {gamma}, alpha: {alpha}, n_models: {n_models}")

    train_predictions, test_predictions = ensemble_kernel_ridge_regression(X_train, y_train, X_test, gamma, alpha, n_models)

    if rank == 0:
        train_rmse = sqrt(mean_squared_error(y_train, train_predictions))
        test_rmse = sqrt(mean_squared_error(y_test, test_predictions))
        print(f"Ensemble Train Root Mean Squared Error: {train_rmse}")
        print(f"Ensemble Test Root Mean Squared Error: {test_rmse}")

if __name__ == "__main__":
    main()