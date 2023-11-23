from pyspark.sql import DataFrame
import pyspark.sql.functions as pf
from common import *
import matplotlib.pyplot as plt
import numpy as np
import math


def dataset_stats(df: DataFrame, output_path: str):
    # Calculate the number of rows for each features
    total_count = df.count()

    print(f"Total data count: {total_count}")

    feature_groups_df = df.groupBy(pf.col(col=COL_WEATHER_CONDITION)) \
        .count()

    counts = {}
    for r in feature_groups_df.collect():
        feature = r[COL_WEATHER_CONDITION]
        count = r["count"]
        counts[feature] = (
            count, f"{round(ndigits=3, number=(count/total_count))*100}%")

    selected_df = df \
        .select(pf.col(COL_TEMPERATURE).cast("double"),
                pf.col(COL_HUMIDITY).cast("double"),
                pf.col(COL_PRESSURE).cast("double"),
                pf.col(COL_WIND_DIRECTION).cast("double"),
                pf.col(COL_WIND_SPEED).cast("double"))

    summ = selected_df.summary()

    summ.show()

    summary_rows = summ.collect()

    fig, ax = plt.subplots(1, 2, figsize=(16, 9))

    # Dataset count by features
    features = []
    feature_data_counts = []
    for f, d in counts.items():
        features.append(f)
        feature_data_counts.append(d[0])

    ax[0].bar(features, feature_data_counts)
    ax[0].set_ylabel("Data count")
    ax[0].set_title("Data count by feature")

    print(counts)

    features = [COL_TEMPERATURE, COL_HUMIDITY,
                COL_PRESSURE, COL_WIND_SPEED]

    features_min = summary_rows[3]
    mins = [features_min[COL_TEMPERATURE], features_min[COL_HUMIDITY],
            features_min[COL_PRESSURE], features_min[COL_WIND_SPEED]]

    features_max = summary_rows[7]
    maxes = [features_max[COL_TEMPERATURE], features_max[COL_HUMIDITY],
             features_max[COL_PRESSURE], features_max[COL_WIND_SPEED]]

    for i in range(len(mins)):
        mins[i] = float(mins[i])

    for i in range(len(maxes)):
        maxes[i] = float(maxes[i])

    print(f"Minimums: {mins}")
    print(f"Maximums: {maxes}")

    rows = df.collect()

    temps = [normalize(float(r[COL_TEMPERATURE]), mins[0], maxes[0])
             for r in rows]
    humidity = [normalize(float(r[COL_HUMIDITY]), mins[1], maxes[1])
                for r in rows]
    pressure = [normalize(float(r[COL_PRESSURE]), mins[2], maxes[2])
                for r in rows]
    wind_speed = [normalize(float(r[COL_WIND_SPEED]),
                            mins[3], maxes[3]) for r in rows]
    boxplot_data = [temps, humidity, pressure, wind_speed]

    # metrics_stddev = rows[2]
    # stddev = [metrics_stddev[COL_TEMPERATURE], metrics_stddev[COL_HUMIDITY],
    #           metrics_stddev[COL_PRESSURE], metrics_stddev[COL_WIND_DIRECTION], metrics_stddev[COL_WIND_SPEED]]

    # print(metrics)
    # print(means)
    # print(stddev)

    ax[1].boxplot(boxplot_data, labels=features)
    ax[1].set_title("Data variance for each features")

    plt.savefig(os.path.join(output_path, "stats"))


def normalize(data, min, max):
    diff = (max - min)
    if diff == 0:
        return 0
    return (data - min) / diff
