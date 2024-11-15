# Weather Pressure Prediction Model
This project implements machine learning models to predict sea level pressure using various weather-related features, running on Google Cloud Dataproc.

## Project Structure
All code and extracted csv files are stored in a bucket called `dsa5208-weather`.
```
gs://dsa5208-weather/
├── code/
│   └── weather.py
└── extracted/
    └── *.csv (weather data files)
```

## Models Implemented & Results
**Note: these results are for entire data sample (>14000 files)**
1. Linear Regression
   - Train RMSE: 86.0902
   - Test RMSE: 86.1117
   - Test R2: 0.0998
   - Test MAE: 62.8752
   - Best Parameters:
     - regParam: 0.0
     - elasticNetParam: 0.0
   - Top Features: wind_speed (-550.39), air_temp (-96.63), dew_point (-99.84)

2. Random Forest
   - Train RMSE: 82.3007
   - Test RMSE: 82.3089
   - Test R2: 0.1776
   - Test MAE: 59.7790
   - Best Parameters:
     - numTrees: 20
     - maxDepth: 5
   - Top Features: latitude (0.22), air_temp (0.19), ceiling_height (0.16)

3. Gradient Boosted Trees
   - Train RMSE: 78.6997
   - Test RMSE: 78.6995
   - Test R2: 0.2481
   - Test MAE: 56.7953
   - Best Parameters:
     - maxDepth: 5
     - maxIter: 20
   - Top Features: latitude (0.20), air_temp (0.16), longitude (0.16)

## Usage
Run directly from terminal **(Run without SSHing)**:
```bash
gcloud dataproc jobs submit pyspark gs://dsa5208-weather/code/weather.py \
    --cluster <your-cluster-name> \
    --region <your-region>
```

## Technical Implementation
- Uses PySpark's Pipeline API for efficient transformations
- Strategic caching with .cache() and .unpersist() for performance
- MinMax scaling implemented (StandardScaler also available)
- 70-30 train-test split with 3-fold cross validation

## Potential Improvements
1. Experiment with different scaling methods:
- Try StandardScaler
- Compare unscaled vs scaled performance

2. Enhanced Feature Engineering:
- Incorporate quality control scores from documentation
- Use quality scores as additional features??
- For example for Air Temperature:
    - 0 = Passed gross limits check
    - 1 = Passed all quality control checks
    - 2 = Suspect
    - 3 = Erroneous
    - 4 = Passed gross limits check, data originate from an NCEI data source
    - 5 = Passed all quality control checks, data originate from an NCEI data source
    - 6 = Suspect, data originate from an NCEI data source
    - 7 = Erroneous, data originate from an NCEI data source
    - 9 = Passed gross limits check if element is present

3. Model Improvements:
- Implement Neural Network model
- Optimize null parsing method at the start
- Expand hyperparameter search grid

## Tips
- Stop/delete other Dataproc jobs before running (Monitor via console.google.com)
- XGBoost and initial data preprocessing may take significant time
