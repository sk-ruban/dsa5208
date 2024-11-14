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
**Note: these results are for the 1000 Files Sample, Not the entire >14000 Files!**
1. Linear Regression
   - Train RMSE: 86.3446
   - Test RMSE: 86.3267
   - Test R2: 0.0881
   - Test MAE: 63.3234
   - Best Parameters:
     - regParam: 0.0
     - elasticNetParam: 0.0
   - Top Features: wind_speed (-413.15), air_temp (-64.16), dew_point (-48.95)

2. Random Forest
   - Train RMSE: 81.5404
   - Test RMSE: 81.4562
   - Test R2: 0.1881
   - Test MAE: 59.5847
   - Best Parameters:
     - numTrees: 20
     - maxDepth: 5
   - Top Features: latitude (0.28), air_temp (0.17), wind_speed (0.13)

3. Gradient Boosted Trees
   - Train RMSE: 77.3908
   - Test RMSE: 77.3523
   - Test R2: 0.2678
   - Test MAE: 55.9982
   - Best Parameters:
     - maxDepth: 5
     - maxIter: 20
   - Top Features: latitude (0.20), dew_point (0.13), air_temp (0.13)

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
