"""
RUN IN TERMINAL WITH: mpiexec -n 4 /usr/local/bin/python3 basic_krr_mpi.py
Training RMSE: 16797.242683154138
Test RMSE: 19656.63335409789
"""

from mpi4py import MPI
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        data = pd.read_csv('housing.tsv', sep='\t')

        data.columns = ['longitude', 'latitude', 'housingMedianAge', 'totalRooms',
              'totalBedrooms', 'population', 'households', 'medianIncome',
              'oceanProximity', 'medianHouseValue']

        X = data.drop(['medianHouseValue', 'oceanProximity'], axis=1)
        y = data['medianHouseValue']
        

        X = pd.get_dummies(data, columns=['oceanProximity'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        chunks = np.array_split(X_train_scaled, size)
        y_chunks = np.array_split(y_train, size)
    else:
        chunks = None
        y_chunks = None

    chunk = comm.scatter(chunks, root=0)
    y_chunk = comm.scatter(y_chunks, root=0)

    local_model = KernelRidge(alpha=1.0, kernel='rbf')
    local_model.fit(chunk, y_chunk)

    models = comm.gather(local_model, root=0)

    if rank == 0:
        def ensemble_predict(X):
            predictions = np.array([model.predict(X) for model in models])
            return np.mean(predictions, axis=0)

        y_train_pred = ensemble_predict(X_train_scaled)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        print(f"Training RMSE: {train_rmse}")

        y_test_pred = ensemble_predict(X_test_scaled)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        print(f"Test RMSE: {test_rmse}")

if __name__ == "__main__":
    main()