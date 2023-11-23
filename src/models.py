from pyspark.ml.tuning import CrossValidatorModel
from pyspark.sql import DataFrame
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import IndexToString
from pyspark.ml import Pipeline
from typing import * 
from encoding import *
import sys
sys.path.append("..")
sys.path.append(".")
from utils import common

def random_forest_pipeline(dataframe: DataFrame,
                           numerical_features: List[str],
                           categorical_features: List[str],
                           target_variable: str,
                           features_col: str,
                           with_std: bool = True,
                           with_mean: bool = False,
                           k_fold: int = 5) -> CrossValidatorModel:
  
    
  
    data_encoder = encoding_pipeline(dataframe,
                                     numerical_features,
                                     categorical_features,
                                     target_variable,
                                     with_std,
                                     with_mean)
  
    classifier = RandomForestClassifier(featuresCol=features_col, labelCol=common.COL_LABEL)
    
    predictions_idx_to_str = IndexToString(inputCol=common.COL_PREDICTION,
                                           outputCol=common.COL_PREDICTED_TARGET_VARIABLE,
                                           labels=data_encoder.stages[0].labels)

    stages = [data_encoder, classifier, predictions_idx_to_str]

    pipeline = Pipeline(stages=stages)

    # We use a ParamGridBuilder to construct a grid of parameters to search over.
    # With 3 values for maxDepth and 3 values for numTrees
    # this grid will have 3 x 3 = 9 parameter settings for CrossValidator to choose from.
    param_grid = ParamGridBuilder() \
        .addGrid(classifier.maxDepth, [3, 5, 8]) \
        .addGrid(classifier.numTrees, [10, 50, 100]) \
        .build()

    evaluator = MulticlassClassificationEvaluator(labelCol=common.COL_LABEL,
                                                  predictionCol=common.COL_PREDICTION,
                                                  metricName='accuracy')

    cross_val = CrossValidator(estimator=pipeline, 
                               estimatorParamMaps=param_grid,
                               evaluator=evaluator,
                               numFolds=k_fold,
                               collectSubModels=True)

    cv_model = cross_val.fit(dataframe)

    return cv_model