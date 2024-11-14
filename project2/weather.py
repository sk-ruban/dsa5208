from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, MinMaxScaler
from pyspark.ml.regression import RandomForestRegressor, LinearRegression, GBTRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, split, count, when, input_file_name, lit
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.stat import Summarizer
import sys

# Reduce logging
spark = SparkSession.builder \
    .appName("Weather Pressure Prediction") \
    .config("spark.ui.showConsoleProgress", "false") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

def load_and_preprocess_data():
    """
    Returns cleaned DataFrame with necessary features
    """
    print("Starting data preprocessing...")

    # USE ONLY IF YOU WANT A SAMPLE (eg. 1000 files) AND COMMENT OUT THE NEXT 3 LINES
    files = spark.sparkContext.binaryFiles("gs://dsa5208-weather/extracted/*.csv") \
                    .map(lambda x: x[0]) \
                    .takeSample(False, 2000, seed=42)

    print(f"\nSelected {len(files)} files")
    selected_columns = ["LATITUDE", "LONGITUDE", "ELEVATION",
                       "WND", "CIG", "VIS", "TMP", "DEW", "SLP"]

    df = spark.read.csv(files,
                       header=True,
                       inferSchema=True) \
             .select(selected_columns) \
             .cache()

    # USE IF WANT TO MODEL ENTIRE DATASET
    # df = spark.read.csv("gs://dsa5208-weather/extracted/*.csv",
    #                     header=True,
    #                     inferSchema=True)

    # selected_columns = ["LATITUDE", "LONGITUDE", "ELEVATION",
    #                    "WND", "CIG", "VIS", "TMP", "DEW", "SLP"]

    # df = df.select([selected_columns]).cache()

    initial_count = df.count()
    print(f"Initial row count: {initial_count}")

    print("\nSample of raw data before parsing:")
    df.show(20, truncate=False)

    print("\nParsing columns...")
    df_parsed = df.select(
        # Latitude: +99999 is missing
        when(col("LATITUDE").contains("+99999"), None)
        .otherwise(col("LATITUDE").cast("double"))
        .alias("latitude"),

        # Longitude: +999999 is missing
        when(col("LONGITUDE").contains("+999999"), None)
        .otherwise(col("LONGITUDE").cast("double"))
        .alias("longitude"),

        # Elevation: +9999 is missing
        when(col("ELEVATION").contains("+9999"), None)
        .otherwise(col("ELEVATION").cast("double"))
        .alias("elevation"),

        # Wind direction: 999 is missing
        when(split(col("WND"), ",").getItem(0) == "999", None)
        .otherwise(split(col("WND"), ",").getItem(0).cast("double"))
        .alias("wind_direction"),

        # Wind speed: 9999 is missing in fourth position
        when(split(col("WND"), ",").getItem(3) == "9999", None)
        .otherwise(split(col("WND"), ",").getItem(3).cast("double"))
        .alias("wind_speed"),

        # Ceiling height: 99999 is missing
        when(split(col("CIG"), ",").getItem(0) == "99999", None)
        .otherwise(split(col("CIG"), ",").getItem(0).cast("double"))
        .alias("ceiling_height"),

        # Visibility: 999999 is missing
        when(split(col("VIS"), ",").getItem(0) == "999999", None)
        .otherwise(split(col("VIS"), ",").getItem(0).cast("double"))
        .alias("visibility"),

        # Air temperature: +9999 is missing
        when(col("TMP").contains("+9999"), None)
        .otherwise(split(col("TMP"), ",").getItem(0).cast("double"))
        .alias("air_temp"),

        # Dew point: +9999 is missing
        when(col("DEW").contains("+9999"), None)
        .otherwise(split(col("DEW"), ",").getItem(0).cast("double"))
        .alias("dew_point"),

        # Sea level pressure: 99999 is missing
        when(split(col("SLP"), ",").getItem(0) == "99999", None)
        .otherwise(split(col("SLP"), ",").getItem(0).cast("double"))
        .alias("sea_level_pressure")
    )

    df.unpersist()
    print("\nSample of parsed data:")
    df_parsed.show(20)

    print("\nNull counts after parsing:")
    for column in df_parsed.columns:
        null_count = df_parsed.filter(col(column).isNull()).count()
        total = df_parsed.count()
        print(f"{column}: {null_count} nulls out of {total} ({(null_count/total)*100:.2f}%)")

    print("\nRemoving rows with null values...")
    df_no_nulls = df_parsed.dropna().cache()

    rows_after_null_removal = df_no_nulls.count()
    print(f"Rows remaining after null removal: {rows_after_null_removal}")

    print("\nApplying valid range filters...")
    df_filtered = df_no_nulls.filter(
        (col("latitude").between(-90000, 90000)) &
        (col("longitude").between(-179999, 180000)) &
        (col("elevation").between(-400, 8850)) &
        (col("wind_direction").between(1, 360)) &
        (col("wind_speed").between(0, 900)) &
        (col("ceiling_height").between(0, 22000)) &
        (col("visibility").between(0, 160000)) &
        (col("air_temp").between(-932, 618)) &
        (col("dew_point").between(-982, 368)) &
        (col("sea_level_pressure").between(8600, 10900))
    )

    df_no_nulls.unpersist()
    final_count = df_filtered.count()
    print(f"Final row count after all filtering: {final_count}")

    return df_filtered

def prepare_features(df):
    """Prepare feature vectors with scaling"""
    print("\nApplying scaling...")

    assembler = VectorAssembler(
       inputCols=[
           "latitude", "longitude", "elevation",
           "wind_direction", "wind_speed",
           "ceiling_height", "visibility",
           "air_temp", "dew_point"
       ],
       outputCol="features"
    )

    assembled_df = assembler.transform(df)

    # YOU CAN USE STANDARD SCALAR TOO
    # standard_scaler = StandardScaler(
    #     inputCol="assembled_features",
    #     outputCol="standard_scaled_features",
    #     withStd=True,
    #     withMean=True
    # )
    #
    minmax_scaler = MinMaxScaler(
        inputCol="features",
        outputCol="scaled_features",
        min=0.0,
        max=1.0
    )

    pipeline = Pipeline(stages=[assembler, minmax_scaler])
    model = pipeline.fit(df)

    return model.transform(df), assembler.getInputCols()

def train_and_evaluate(scaled_df, feature_names):
    """Train and evaluate models"""
    print("\nStarting model training and evaluation...")

    train_data, test_data = scaled_df.randomSplit([0.7, 0.3], seed=42)
    train_data = train_data.cache()
    test_data = test_data.cache()
    print(f"Training set size: {train_data.count()}")
    print(f"Test set size: {test_data.count()}")

    try:
        evaluator = RegressionEvaluator(
            labelCol="sea_level_pressure",
            predictionCol="prediction"
        )

        models = {
            "Linear Regression": (
                LinearRegression(standardization=False),
                ParamGridBuilder()
                    .addGrid(LinearRegression.regParam, [0.01, 0.1, 1.0])
                    .addGrid(LinearRegression.elasticNetParam, [0.0, 0.5, 1.0])
                    .build()
            ),

            "Random Forest": (
                RandomForestRegressor(),
                ParamGridBuilder()
                    .addGrid(RandomForestRegressor.numTrees, [20, 40, 60])
                    .addGrid(RandomForestRegressor.maxDepth, [4, 8, 12])
                    .build()
            ),

            "Gradient Boosted Trees": (
                GBTRegressor(),
                ParamGridBuilder()
                    .addGrid(GBTRegressor.maxDepth, [4, 8])
                    .addGrid(GBTRegressor.maxIter, [20, 40])
                    .build()
            )
        }

        print("\nTraining models...")

        for model_name, (model, param_grid) in models.items():
            print(f"\nTraining {model_name}...")

            model.setFeaturesCol("scaled_features")
            model.setLabelCol("sea_level_pressure")

            cv = CrossValidator(
                estimator=model,
                estimatorParamMaps=param_grid,
                evaluator=evaluator,
                numFolds=3
            )

            cv_model = cv.fit(train_data)
            best_model = cv_model.bestModel

            train_predictions = best_model.transform(train_data)
            test_predictions = best_model.transform(test_data)

            evaluator.setMetricName("rmse")
            train_rmse = evaluator.evaluate(train_predictions)
            test_rmse = evaluator.evaluate(test_predictions)

            evaluator.setMetricName("r2")
            test_r2 = evaluator.evaluate(test_predictions)

            evaluator.setMetricName("mae")
            test_mae = evaluator.evaluate(test_predictions)

            print(f"\n{model_name} Results:")
            print(f"Train RMSE: {train_rmse:.4f}")
            print(f"Test RMSE: {test_rmse:.4f}")
            print(f"Test R2: {test_r2:.4f}")
            print(f"Test MAE: {test_mae:.4f}")

            if model_name == "Linear Regression":
                print(f"\nBest Parameters:")
                print(f"regParam: {best_model.getRegParam()}")
                print(f"elasticNetParam: {best_model.getElasticNetParam()}")

                coefficients = best_model.coefficients.toArray()
                print("\nTop 5 most important features by absolute coefficient value:")
                coef_pairs = [(name, coef) for name, coef in zip(feature_names, coefficients)]
                sorted_coefs = sorted(coef_pairs, key=lambda x: abs(x[1]), reverse=True)
                for name, coef in sorted_coefs[:5]:
                    print(f"{name}: {coef:.4f}")

            elif model_name in ["Random Forest", "Gradient Boosted Trees"]:
                print(f"\nBest Parameters:")
                if model_name == "Random Forest":
                   print(f"numTrees: {best_model.getNumTrees()}")
                   print(f"maxDepth: {best_model.getMaxDepth()}")
                else:
                   print(f"maxDepth: {best_model.getMaxDepth()}")
                   print(f"maxIter: {best_model.getMaxIter()}")

                if hasattr(best_model, 'featureImportances'):
                    print("\nTop 5 features by importance:")
                    importances = best_model.featureImportances
                    feature_imp = [(feat, float(imp)) for feat, imp in zip(feature_names, importances)]
                    sorted_imp = sorted(feature_imp, key=lambda x: x[1], reverse=True)[:5]
                    for feat, imp in sorted_imp:
                        print(f"{feat}: {imp:.4f}")

    except Exception as e:
        print(f"Error during model training: {str(e)}")
        raise
    finally:
        print("\nCleaning up cached DataFrames...")
        train_data.unpersist()
        test_data.unpersist()

def main():
    df = load_and_preprocess_data()
    scaled_df, feature_names = prepare_features(df)
    train_and_evaluate(scaled_df, feature_names)
    print("\nProcessing complete!")
    spark.stop()

if __name__ == "__main__":
    main()
