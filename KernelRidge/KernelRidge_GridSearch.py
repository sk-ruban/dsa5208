from mpi4py import MPI
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Load and preprocess data
        data = pd.read_csv('data/housing.tsv', sep='\t')

        data.columns = ['longitude', 'latitude', 'housingMedianAge', 'totalRooms',
              'totalBedrooms', 'population', 'households', 'medianIncome',
              'oceanProximity', 'medianHouseValue']

        X = data.drop('medianHouseValue', axis=1)
        y = data['medianHouseValue']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    else:
        X_train, X_test, y_train, y_test = None, None, None, None

    # Broadcast the data to all processes
    X_train = comm.bcast(X_train, root=0)
    y_train = comm.bcast(y_train, root=0)
    X_test = comm.bcast(X_test, root=0)
    y_test = comm.bcast(y_test, root=0)

    # Create a preprocessing pipeline
    numeric_features = ['longitude', 'latitude', 'housingMedianAge', 'totalRooms',
                        'totalBedrooms', 'population', 'households', 'medianIncome']
    categorical_features = ['oceanProximity']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', 'passthrough', categorical_features)
        ])

    # Create a pipeline with preprocessor and KernelRidge
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', KernelRidge(kernel='rbf'))
    ])

    # Define parameter grid for search
    param_grid = {
        'regressor__alpha': [0.1, 1.0, 10.0],
        'regressor__gamma': [0.1, 1.0, 10.0]
    }

    # Perform grid search with cross-validation
    print("Performing Grid SEARCH")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get best model
    best_model = grid_search.best_estimator_

    # Make predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Calculate RMSE (only rank 0 will print)
    if rank == 0:
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Training RMSE: {train_rmse}")
        print(f"Test RMSE: {test_rmse}")

if __name__ == "__main__":
    main()