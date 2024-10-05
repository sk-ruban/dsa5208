# Distributed Kernel Ridge Regression with MPI

This project implements a parallel Kernel Ridge Regression algorithm using MPI (Message Passing Interface) for predicting housing prices based on the California Housing dataset.

## Environment Setup

1. Ensure you have Python 3.x installed.
2. Install the required packages:

```bash
python -m pip install mpi4py pandas numpy scikit-learn
```

## Dataset

The project uses the California Housing dataset (`housing_20k.tsv`), which contains 20,000 samples. Each sample has 9 features and 1 target variable (median house value).

## Running the Main Script

To run the main MPI Kernel Ridge Regression script:

```bash
mpiexec -n 8 /usr/local/bin/python3 mpi-kernel.py
```

Note: The number after `-n` specifies the number of processes. Since we're using a dataset of 20,000 samples, choose a number that divides 20,000 evenly, such as 8 or 4.

## Customisation Options

1. **Data Scaling**: You can change the type of scaler used in the `prepare_data()` function. Current options include:
   - StandardScaler (default)
   - MinMaxScaler
   - RobustScaler

   To change, modify line 52 in the `prepare_data()` function.

2. **Kernel Selection**: In the `main()` function, you can choose from the following kernels:
   - Gaussian Kernel (default)
   - Linear Kernel
   - Polynomial Kernel
   - Sigmoid Kernel

   To change the kernel or its parameters, modify the `kernel_func` and `kernel_params` variables in the `main()` function.

## Additional Scripts

1. **Data Analysis**: There's a Jupyter notebook `data_analysis.ipynb` that contains initial data analysis. You can run this to get insights into the dataset.

2. **Cross-Validation and Hyperparameter Testing**: Use the `mpi-cv.py` script for cross-validation and hyperparameter tuning:

```bash
mpiexec -n 8 /usr/local/bin/python3 mpi-cv.py
```

## Project Structure

- `mpi-final.py`: Main script for MPI Kernel Ridge Regression
- `mpi-cv.py`: Script for cross-validation and hyperparameter tuning
- `data_analysis.ipynb`: Jupyter notebook for initial data analysis
- `data/housing_20k.tsv`: Dataset file

## Results

After running the script, you'll see output including:
- Kernel computation time
- Conjugate gradient solver progress
- Training and Test RMSE (Root Mean Square Error)

## Notes

- The project uses a conjugate gradient method to solve the linear system in Kernel Ridge Regression, which is particularly useful for large datasets.
- The implementation includes various kernels and allows for easy switching between them.
- Parallelization is achieved through MPI, which distributes the computation across multiple processes, potentially speeding up the calculation for large datasets.

## Future Improvements

- Implement more advanced hyperparameter tuning methods
- Add support for larger datasets
- Optimize MPI communication patterns for better scalability