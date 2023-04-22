import argparse
import logging

import mlflow
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from gp_classifier.data.half_moon import get_half_moon_data
from gp_classifier.model import get_gpc_model
from gp_classifier.utils import eval_metrics

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Train a sklearn model and log it with mlflow"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-path",
        type=str,
        help="Name of mlflow artifact path location to drop model.",
    )
    parser.add_argument("--tracking-uri", type=str, help="mlflow host and port.")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="my-experiment",
        help="mlflow experiment name",
    )
    parser.add_argument(
        "--data-path", type=str, default="data/historical_data.csv", help="path"
    )
    parser.add_argument(
        "--data-repo",
        type=str,
        default=None,
        help="Location of the DVC project or Git repo",
    )
    parser.add_argument("--data-version", type=str, default="v2", help="version")
    parser.add_argument(
        "--test-size", type=float, default=0.3, help="Fraction of test split."
    )
    parser.add_argument(
        "--length-scale", type=float, default=1.0, help="Length scale RBF"
    )
    parser.add_argument(
        "--n-jobs",
        type=float,
        default=2,
        help="Number of jobs to compute multiclass problems in parallel",
    )
    parser.add_argument("--seed", type=str, help="seed experiment")
    args = parser.parse_args()

    artifact_path = args.artifact_path
    tracking_uri = args.tracking_uri
    experiment_name = args.experiment_name
    data_path = args.data_path
    data_repo = args.data_repo
    data_version = args.data_version
    test_size = args.test_size
    length_scale = args.length_scale
    n_jobs = args.n_jobs
    seed = args.seed

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    mlflow.end_run()

    # Get train and test data
    X_train, X_test, y_train, y_test = get_half_moon_data(
        data_path=data_path,
        data_repo=data_repo,
        data_version=data_version,
        split="all",
        test_size=test_size,
        seed=seed,
    )

    # Create GPC sklearn model pipeline
    model = get_gpc_model(length_scale=length_scale, n_jobs=n_jobs, seed=seed)
    model.fit(X_train, y_train)

    # Compute metrics on test set
    test_preds, _ = model.predict(X_test)
    metrics = eval_metrics(y_test, test_preds)
    acc = metrics.get("acc")
    f1 = metrics.get("f1")
    recall = metrics.get("recall")
    precision = metrics.get("precision")

    acc = metrics.get("acc")
    logger.info(f"Acc: {acc:2f}")
    logger.info(f"f1: {f1:2f}")
    logger.info(f"recall: {recall:2f}")
    logger.info(f"precision: {precision:2f}")

    # Start a run in the experiment and save and register the model and metrics
    with mlflow.start_run() as run:
        run_num = run.info.run_id
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=run_num, artifact_path=artifact_path
        )

        # Log dataset
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_samples", X_train.shape[0])
        mlflow.log_param("version", data_version)
        mlflow.log_param("seed", seed)
        mlflow.log_param("n_train", X_train.shape[0])
        mlflow.log_param("n_test", X_test.shape[0])
        mlflow.log_param("test_size", test_size)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)

        mlflow.sklearn.log_model(model, artifact_path)
        mlflow.log_params(model.get_params())

        mlflow.register_model(model_uri=model_uri, name=artifact_path)

        # Compute confusion matrix
        np.set_printoptions(precision=2)

        disp = ConfusionMatrixDisplay.from_predictions(
            y_true=y_test,
            y_pred=test_preds,
            normalize="true",
        )
        mlflow.log_dict(
            np.array(disp.confusion_matrix).tolist(),
            artifact_file=f"{artifact_path}/metrics/confusion_matrix.json",
        )


if __name__ == "__main__":
    main()
