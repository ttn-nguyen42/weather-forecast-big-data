import os
from pyspark.sql import SparkSession, DataFrame

DATASET_PATH = "./dataset"

# Datasets
DS_WEATHER_DESCRIPTION = os.path.join(DATASET_PATH, "weather_description.csv")
DS_WIND_DIRECTION = os.path.join(DATASET_PATH, "wind_direction.csv")
DS_WIND_SPEED = os.path.join(DATASET_PATH, "wind_speed.csv")
DS_TEMPERATURE = os.path.join(DATASET_PATH, "temperature.csv")
DS_PRESSURE = os.path.join(DATASET_PATH, "pressure.csv")
DS_HUMIDITY = os.path.join(DATASET_PATH, "humidity.csv")
DS_CITY_ATTRIBUTES = os.path.join(DATASET_PATH, "city_attributes.csv")

# Columns
COL_DATETIME = "datetime"

#   Features
COL_FEATURES = "features"
COL_SCALED_FEATURES = f"scaled_{COL_FEATURES}"

#   Numerical features
COL_HUMIDITY = "humidity"
COL_PRESSURE = "pressure"
COL_TEMPERATURE = "temperature"
COL_WIND_DIRECTION = "wind_direction"
COL_WIND_SPEED = "wind_speed"
COL_LATITUDE = "latitude"
COL_LONGITUDE = "longitude"
NUMERICAL_FEATURES = [
    COL_HUMIDITY, 
    COL_PRESSURE,
    COL_TEMPERATURE,
    COL_WIND_DIRECTION,
    COL_WIND_SPEED,
    COL_LATITUDE,
    COL_LONGITUDE
]

COL_CITY = "city"
COL_COUNTRY = "country"
COL_WEATHER_CONDITION = "weather_condition"
COL_LABEL = "label"
COL_PREDICTION = "prediction"
COL_PREDICTED_TARGET_VARIABLE = f"predicted_{COL_WEATHER_CONDITION}"

# Preprocessing
PREPROCESS_PATH = "prep"
PRE_PIPELINE_OUTPUT = os.path.join(PREPROCESS_PATH, "outputs")
PRE_FINAL_CSV = os.path.join(PREPROCESS_PATH, "dataset.csv")
PRE_SAMPLED_CSV = os.path.join(PREPROCESS_PATH, "sampled.csv")
PRE_TRAINING_CSV = os.path.join(PREPROCESS_PATH, "training.csv")
PRE_TESTING_CSV = os.path.join(PREPROCESS_PATH, "testing.csv")

SPLIT_RATIO = [0.8, 0.2]

#   Categorical features
COND_RAINY = "rainy"
COND_SNOWY = "snowy"
COND_SUNNY = "sunny"
COND_FOGGY = "foggy"
COND_CLOUDY = "cloudy"
COND_THUNDERSTORM = "thunderstorm"

WEATHER_CONDITIONS = [
    COND_RAINY,
    COND_SNOWY,
    COND_SUNNY,
    COND_FOGGY,
    COND_CLOUDY,
    COND_THUNDERSTORM,
]


def load_df(session: SparkSession, path: str) -> DataFrame:
    return session.read \
        .format(source="csv") \
        .option("header", True) \
        .option("delimiter", ",") \
        .load(path=path)


def combine_csv(input_dir: str, output_file_path: str):
    import pandas as pd
    import os

    all_files = os.listdir(path=input_dir)
    csv_files = [f for f in all_files if f.endswith(".csv")]

    df_list = []

    for csv in csv_files:
        file_path = os.path.join(input_dir, csv)
        try:
            # Try reading the file using default UTF-8 encoding
            df = pd.read_csv(file_path)
            df_list.append(df)
        except UnicodeDecodeError:
            try:
                # If UTF-8 fails, try reading the file using UTF-16 encoding with tab separator
                df = pd.read_csv(file_path, sep='\t', encoding='utf-16')
                df_list.append(df)
            except Exception as e:
                print(f"Could not read file {csv} because of error: {e}")
        except Exception as e:
            print(f"Could not read file {csv} because of error: {e}")

    # Concatenate all data into one DataFrame
    big_df = pd.concat(df_list, ignore_index=True)

    # Save the final result to a new CSV file
    big_df.to_csv(os.path.join(output_file_path), index=False)


def to_aggr_condition(condition: str) -> str:
    lower = condition.lower()

    if is_thunderstorm(cond=lower):
        return COND_THUNDERSTORM
    if is_rainy(cond=lower):
        return COND_RAINY
    if is_snowy(cond=lower):
        return COND_SNOWY
    if is_cloudy(cond=lower):
        return COND_CLOUDY
    if is_foggy(cond=lower):
        return COND_FOGGY
    if is_sunny(cond=lower):
        return COND_SUNNY

    return ""


def is_thunderstorm(cond: str) -> bool:
    if "squall" in cond:
        return True
    if "thunderstorm" in cond:
        return True
    return False


def is_rainy(cond: str) -> bool:
    if "drizzle" in cond:
        return True
    if "rain" in cond:
        return True
    return False


def is_snowy(cond: str) -> bool:
    if "sleet" in cond:
        return True
    if "snow" in cond:
        return True
    return False


def is_cloudy(cond: str) -> bool:
    if "cloud" in cond:
        return True
    return False


def is_foggy(cond: str) -> bool:
    if "fog" in cond:
        return True
    if "mist" in cond:
        return True
    if "haze" in cond:
        return True
    return False


def is_sunny(cond: str) -> bool:
    if "clear" in cond:
        return True
    if "sun" in cond:
        return True
    return False
