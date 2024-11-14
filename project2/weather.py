from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, split, count, when, input_file_name, lit
from pyspark.sql.types import *
import sys

# Reduced logging
spark = SparkSession.builder \
    .appName("Weather Pressure Prediction") \
    .config("spark.ui.showConsoleProgress", "false") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

def print_progress(message):
    """Print progress message on same line"""
    sys.stdout.write(f"\r{message}")
    sys.stdout.flush()

def load_and_preprocess_data():
    """
    Returns cleaned DataFrame with necessary features
    """
    print("Starting data preprocessing...")
    print("\nGetting file list...")
    files_df = spark.read.format("binaryFile") \
                   .load("gs://dsa5208-weather/extracted/*.csv") \
                   .select("path") \
                   .orderBy("path") \
                   .limit(20)

    file_paths = [row.path for row in files_df.collect()]
    print("\nSelected files:")
    for path in file_paths:
        print(path)

    print("\nReading selected columns from files...")
    selected_columns = ["LATITUDE", "LONGITUDE", "ELEVATION",
                       "WND", "CIG", "VIS", "TMP", "DEW", "SLP"]

    df = spark.read.csv(file_paths,
                       header=True,
                       inferSchema=True) \
               .select(selected_columns)

    initial_count = df.count()
    print(f"\nInitial row count: {initial_count}")

    # Show data sample
    print("\nSample of raw data before parsing:")
    df.show(20, truncate=False)

    print("\nDataframe schema:")
    df.printSchema()

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

    print("\nSample of parsed data:")
    df_parsed.show(20)

    # Print null counts for each column
    print("\nNull counts after parsing:")
    for column in df_parsed.columns:
        null_count = df_parsed.filter(col(column).isNull()).count()
        total = df_parsed.count()
        print(f"{column}: {null_count} nulls out of {total} ({(null_count/total)*100:.2f}%)")

    # Remove rows with any null values
    print("\nRemoving rows with null values...")
    df_no_nulls = df_parsed.dropna()
    rows_after_null_removal = df_no_nulls.count()
    print(f"Rows remaining after null removal: {rows_after_null_removal}")

    # Apply valid range filters based on documentation
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

    final_count = df_filtered.count()
    print(f"\nFinal row count after all filtering: {final_count}")

    return df_filtered

def prepare_features(df):
    """Prepare feature vector for ML"""
    print("Preparing features...")

    assembler = VectorAssembler(
        inputCols=[
            "latitude", "longitude", "elevation",
            "wind_direction", "wind_speed",
            "ceiling_height", "visibility",
            "air_temp", "dew_point"
        ],
        outputCol="features"
    )

    return assembler.transform(df)

def train_and_evaluate(df):
    print_progress("Training and evaluating models...")

    # Split data
    train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)
    print(f"\nTraining set size: {train_data.count()}")
    print(f"Test set size: {test_data.count()}")

    # Train Random Forest
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="sea_level_pressure",
        numTrees=100,
        maxDepth=10,
        seed=42
    )
    rf_model = rf.fit(train_data)

    # Train Linear Regression
    lr = LinearRegression(
        featuresCol="features",
        labelCol="sea_level_pressure",
        maxIter=100
    )
    lr_model = lr.fit(train_data)

    # Evaluate models
    evaluator = RegressionEvaluator(labelCol="sea_level_pressure", predictionCol="prediction")

    # Random Forest evaluation
    rf_predictions = rf_model.transform(test_data)
    rf_rmse = evaluator.setMetricName("rmse").evaluate(rf_predictions)
    rf_r2 = evaluator.setMetricName("r2").evaluate(rf_predictions)
    rf_mae = evaluator.setMetricName("mae").evaluate(rf_predictions)

    print("\nRandom Forest Performance:")
    print(f"RMSE: {rf_rmse:.4f}")
    print(f"MAE: {rf_mae:.4f}")
    print(f"R2: {rf_r2:.4f}")

    # Linear Regression evaluation
    lr_predictions = lr_model.transform(test_data)
    lr_rmse = evaluator.setMetricName("rmse").evaluate(lr_predictions)
    lr_r2 = evaluator.setMetricName("r2").evaluate(lr_predictions)
    lr_mae = evaluator.setMetricName("mae").evaluate(lr_predictions)

    print("\nLinear Regression Performance:")
    print(f"RMSE: {lr_rmse:.4f}")
    print(f"MAE: {lr_mae:.4f}")
    print(f"R2: {lr_r2:.4f}")

def main():
    print("Starting weather prediction model...")

    # Load and preprocess data
    df = load_and_preprocess_data()

    # Prepare features
    df_prepared = prepare_features(df)

    # Train and evaluate models
    train_and_evaluate(df_prepared)

    print("\nProcessing complete!")
    spark.stop()

if __name__ == "__main__":
    main()
