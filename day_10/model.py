import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_digits
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    acc = accuracy_score(actual, pred)
    prec = precision_score(actual, pred, average = 'micro')
    rec = recall_score(actual, pred, average = 'micro')
    return acc, prec, rec


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    digits = load_digits(as_frame=True)

    X = digits['data']
    y = digits['target']

    X_train, X_test, y_train, y_test = train_test_split(X,y)

    C = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    penalty = sys.argv[2] if len(sys.argv) > 2 else 'l2'

    with mlflow.start_run():
        lr = LogisticRegression(C=C, penalty=penalty, random_state=42)
        lr.fit(X_train, y_train)

        predicted_digits = lr.predict(X_test)

        (acc, prec, rec) = eval_metrics(y_test, predicted_digits)

        print("LogisticRegression model (C=%f, penalty=%s):" % (C, penalty))
        print("  Acc: %s" % acc)
        print("  Precision: %s" % prec)
        print("  Recall: %s" % rec)

        mlflow.log_param("C", C)
        mlflow.log_param("penalty", penalty)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
        else:
            mlflow.sklearn.log_model(lr, "model")
