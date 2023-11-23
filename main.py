import argparse
import numpy as np
import matplotlib.pyplot as plt 
import os
import  databricks.koalas as ks
from src import models
from utils import common
from utils.eval_helper import * 

from pyspark.ml.feature import IndexToString
def parse_args():
    parser =  argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="prep/test.csv")
    parser.add_argument("--train-data", type=str, default="prep/training.csv")
    parser.add_argument("--test-data", type=str, default="prep/test.csv")

    args = parser.parse_args() 
    return args


def main(args):
    """
    @args: argument from the command line
        @method: method used for classification: [RandomForest, SoftmaxRegression, Deep-learning]
        @train-data: path to trainining dataset (.csv format). This file is obtained by preprocessing the 
                     raw dataset with `preprocessing.py`
        @test-data: path to testing dataset (.csv format). This file is obtained by preprocessing the 
                     raw dataset with `preprocessing.py`
    """ 
    method_names = ["random-forest", "softmax-classification", "mlp"]

    # Create train and test dataset
    train_df = ks.read_csv(args.train_data).to_spark()
    test_df = ks.read_csv(args.test_data).to_spark()

    #   Training (Fitting data)
    assert args.method in method_names, f"Method must be in {method_names}, but found {args.method}" 

    if args.method == "random-forest":
        pipeline, data_encoder = models.random_forest_pipeline(
            dataframe=train_df,
            numerical_features=common.NUMERICAL_FEATURES,
            categorical_features=[],
            target_variable=common.COL_WEATHER_CONDITION,
            features_col=common.COL_FEATURES,
            k_fold=5
        )
 
    if args.method == "softmax-classification":
        pass

    if args.method == "mlp":
        pass

    #   Testing (predictions)
    # encoded_test_data = data_encoder.transform(test_df)    
    predictions = pipeline.transform(test_df)

    # predictions_idx_to_str = IndexToString(inputCol=common.COL_PREDICTION,
    #                                    outputCol=common.COL_PREDICTED_TARGET_VARIABLE,
    #                                    labels=data_encoder.stages[0].labels)
    # predictions_str = predictions_idx_to_str.transform(predictions)
    eval_predictions(predictions) 

if __name__ == '__main__':
    args = parse_args()
    main(args)