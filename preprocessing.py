import shutil
from common import *
from pyspark.sql import SparkSession, DataFrame, Row
import pyspark.sql.functions as pf
import numpy as np
import time
from evaluation import dataset_stats

session: SparkSession = SparkSession.builder \
    .appName("Dataset Preprocessor") \
    .master("local[*]") \
    .getOrCreate()

session.sparkContext.setLogLevel(logLevel="WARN")


def run_prep(city: str):
    # Weather descriptions
    df_description: DataFrame = load_df(
        session=session,
        path=DS_WEATHER_DESCRIPTION
    )

    # Filters out weather descriptions for a specific CITY
    df_description_rows = df_description \
        .withColumn(colName=COL_WEATHER_CONDITION, col=pf.col(col=city)) \
        .select([COL_DATETIME, COL_WEATHER_CONDITION])

    # City location
    df_city_attributes: DataFrame = load_df(
        session=session,
        path=DS_CITY_ATTRIBUTES
    )

    df_city_attributes_row = df_city_attributes \
        .filter(pf.col("City") == city) \
        .select(
            pf.col("City").alias(COL_CITY),
            pf.col("Country").alias(COL_COUNTRY),
            pf.col("Latitude").alias(COL_LATITUDE),
            pf.col("Longitude").alias(COL_LONGITUDE)
        )

    city_attrs = df_city_attributes_row.first().asDict(recursive=True)

    # Temperature
    df_temps: DataFrame = load_df(
        session=session,
        path=DS_TEMPERATURE
    )

    df_temps_rows = df_temps \
        .withColumn(colName=COL_TEMPERATURE, col=pf.col(col=city)) \
        .select([COL_DATETIME, COL_TEMPERATURE])

    # Humidity
    df_humidity: DataFrame = load_df(
        session=session,
        path=DS_HUMIDITY
    )

    df_humidity_rows = df_humidity \
        .withColumn(colName=COL_HUMIDITY, col=pf.col(col=city)) \
        .select([COL_DATETIME, COL_HUMIDITY])

    # Pressure
    df_pressure: DataFrame = load_df(
        session=session,
        path=DS_PRESSURE
    )

    df_pressure_rows = df_pressure \
        .withColumn(colName=COL_PRESSURE, col=pf.col(col=city)) \
        .select([COL_DATETIME, COL_PRESSURE])

    # Wind direction
    df_wind_dir: DataFrame = load_df(
        session=session,
        path=DS_WIND_DIRECTION
    )

    df_wind_dir_rows = df_wind_dir \
        .withColumn(colName=COL_WIND_DIRECTION, col=pf.col(col=city)) \
        .select([COL_DATETIME, COL_WIND_DIRECTION])

    # Wind speed
    df_wind_speed: DataFrame = load_df(
        session=session,
        path=DS_WIND_SPEED
    )

    df_wind_speed_rows = df_wind_speed \
        .withColumn(colName=COL_WIND_SPEED, col=pf.col(col=city)) \
        .select([COL_DATETIME, COL_WIND_SPEED])

    # Joins
    joined_df = df_description_rows \
        .join(df_temps_rows, [COL_DATETIME]) \
        .join(df_humidity_rows, [COL_DATETIME]) \
        .join(df_pressure_rows, [COL_DATETIME]) \
        .join(df_wind_dir_rows, [COL_DATETIME]) \
        .join(df_wind_speed_rows, [COL_DATETIME]) \
        .withColumn(COL_CITY, pf.lit(col=city_attrs[COL_CITY])) \
        .withColumn(COL_COUNTRY, pf.lit(col=city_attrs[COL_COUNTRY])) \
        .withColumn(COL_LATITUDE, pf.lit(col=city_attrs[COL_LATITUDE])) \
        .withColumn(COL_LONGITUDE, pf.lit(col=city_attrs[COL_LONGITUDE]))

    def map_condition(r: Row) -> Row:
        row_dict = r.asDict(recursive=True)
        row_dict.update({COL_WEATHER_CONDITION: to_aggr_condition(
            condition=row_dict[COL_WEATHER_CONDITION])})
        return Row(**row_dict)

    # Reorder columns
    joined_df = joined_df.select([
        COL_DATETIME,
        COL_CITY,
        COL_COUNTRY,
        COL_LATITUDE,
        COL_LONGITUDE,
        COL_TEMPERATURE,
        COL_HUMIDITY,
        COL_PRESSURE,
        COL_WIND_DIRECTION,
        COL_WIND_SPEED,
        COL_WEATHER_CONDITION,
    ]) \
        .dropna() \
        .rdd \
        .map(f=map_condition) \
        .filter(f=lambda r: len(r[COL_WEATHER_CONDITION]) > 0) \
        .toDF()

    ws = joined_df.write \
        .format("csv") \
        .option("header", True) \
        .option("delimiter", ",") \
        .mode("overwrite") \
        .csv(path=PRE_PIPELINE_OUTPUT(city.lower()))

    # Combines CSV
    combine_csv(
        input_dir=PRE_PIPELINE_OUTPUT(city.lower()),
        output_file_path=PRE_FINAL_CSV(city.lower()),
    )

    shutil.rmtree(path=PRE_PIPELINE_OUTPUT(city.lower()))

    dataset_stats(df=joined_df, output_path=PRE_STATISTICS_PATH(city.lower()))

    counts = {}
    # Undersampling
    for cond in WEATHER_CONDITIONS:
        count = joined_df.filter(condition=pf.col(col=COL_WEATHER_CONDITION) == cond) \
            .count()
        counts[cond] = count

    list_counts = counts.values()

    # Some places have no occurence of some weather conditions
    least_amount = np.min([c for c in list_counts if c > 0])

    proportions = {}
    for cond, count in counts.items():
        if count == 0:
            continue
        proportions[cond] = least_amount / count

    print(proportions)

    # Create undersampled dataset
    full_ds = load_df(
        session=session,
        path=PRE_FINAL_CSV(city.lower())
    )

    sampled_df = full_ds.sampleBy(
        col=COL_WEATHER_CONDITION,
        fractions=proportions,
        seed=round(time.time())
    )

    ws = sampled_df.write \
        .format("csv") \
        .option("header", True) \
        .option("delimiter", ",") \
        .mode("overwrite") \
        .csv(path=PRE_PIPELINE_OUTPUT(city.lower()))

    # Combines CSV
    combine_csv(
        input_dir=PRE_PIPELINE_OUTPUT(city.lower()),
        output_file_path=PRE_SAMPLED_CSV(city.lower()),
    )

    shutil.rmtree(path=PRE_PIPELINE_OUTPUT(city.lower()))

    train_df, test_df = sampled_df.randomSplit(
        SPLIT_RATIO, seed=round(time.time()))

    print(f"Training set size = {train_df.count()}")
    print(f"Testing set size = {test_df.count()}")

    # Save these two datasets into a CSV
    ws = train_df.write \
        .format("csv") \
        .option("header", True) \
        .option("delimiter", ",") \
        .mode("overwrite") \
        .csv(path=PRE_PIPELINE_OUTPUT(city.lower()))

    combine_csv(
        input_dir=PRE_PIPELINE_OUTPUT(city.lower()),
        output_file_path=PRE_TRAINING_CSV(city.lower()),
    )

    shutil.rmtree(path=PRE_PIPELINE_OUTPUT(city.lower()))

    ws = test_df.write \
        .format("csv") \
        .option("header", True) \
        .option("delimiter", ",") \
        .mode("overwrite") \
        .csv(path=PRE_PIPELINE_OUTPUT(city.lower()))

    # Combines CSV
    combine_csv(
        input_dir=PRE_PIPELINE_OUTPUT(city.lower()),
        output_file_path=PRE_TESTING_CSV(city.lower()),
    )

    shutil.rmtree(path=PRE_PIPELINE_OUTPUT(city.lower()))


df_city_attributes: DataFrame = load_df(
    session=session,
    path=DS_CITY_ATTRIBUTES
)

cities = df_city_attributes.select("City") \
    .collect()

for r in cities:
    print(r["City"])
    run_prep(city=r["City"])

session.stop()
