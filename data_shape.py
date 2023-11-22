from common import *
from pyspark.sql import SparkSession, DataFrame
import logging

session: SparkSession = SparkSession.builder \
    .appName("Dataset Explorer") \
    .master("local[*]") \
    .getOrCreate()

session.sparkContext.setLogLevel(logLevel="FATAL")

df_wd: DataFrame = session.read \
    .format(source="csv") \
    .option("header", True) \
    .option("delimiter", ",") \
    .load(path=DS_WEATHER_DESCRIPTION)

for row in df_wd.head(n=5):
    print(row)

df_temp: DataFrame = session.read \
    .format(source="csv") \
    .option("header", True) \
    .option("delimiter", ",") \
    .load(path=DS_TEMPERATURE)

for row in df_temp.head(n=5):
    print(row)

session.stop()
