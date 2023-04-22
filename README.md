### How to get started


1. Install dependencies

```
    pip install -r requirements.txt
```

2. Pull data with dvc
```
    dvc pull
```

3. Start mlflow server

```
    ./run_server.sh
```

4. Train a `sklearn.GaussianProcessClassifier` and log the model with mlflow

```
    python -m gp_classifier.train --artifact-path gpc --tracking-uri "http://localhost:5000"
```

5. Serve the model

```
    ./serve_model.sh 'models:/gpc/1'
```

Example request of running predictions on latest data

```
    curl http://localhost:8000/invocations  -H 'Content-Type: text/csv' --data-binary @data/latest_data.csv
```

or loading the model from the local store and making the predictions

```
    python -m gp_classifier.predict --model-name gpc --model-version 1
```

### How to run the tests

```
    pytest
```