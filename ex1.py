import mlflow
# from mlflow import log_artifacts, log_metric, log_param

mlflow.set_tracking_uri("http://127.0.0.1:7000")
mlflow.set_experiment("ex1")

with mlflow.start_run():
    mlflow.log_metric("foo", 1)
    mlflow.log_metric("bar", 2)