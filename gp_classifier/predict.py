import argparse
import logging
import os

import mlflow
import pandas as pd

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def main(model_name, model_version, data_path):

    logged_model = f"models:/{model_name}/{model_version}"
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    df = pd.read_csv(data_path, sep=",")

    return loaded_model.predict(df)


if __name__ == "__main__":
    """Make predictions with loaded mlflow-logged model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-path",
        type=str,
        default="./mlflow-artifact-root",
        help="Name of mlflow artifact path location to drop model.",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="http://0.0.0.0:5000",
        help="mlflow host and port.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpc",
        help="Model name as stored on mlflow",
    )
    parser.add_argument("--model-version", type=str, default="1", help="Model version")
    parser.add_argument(
        "--data-path", type=str, default="data/latest_data.csv", help="path"
    )
    args = parser.parse_args()

    os.environ["MLFLOW_TRACKING_URI"] = args.tracking_uri
    os.environ["MLFLOW_ARTIFACT_URI"] = args.artifact_path

    result = main(
        model_name=args.model_name,
        model_version=args.model_version,
        data_path=args.data_path,
    )

    logger.info(result)
