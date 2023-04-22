import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import RBF
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def predict_with_confidence(func):
    def wrapper(*args, **kwargs):
        targets_probs = func(*args, **kwargs)

        targets_preds = np.argmax(targets_probs, axis=1)
        preds_probs = np.amax(targets_probs, axis=1)

        return targets_preds, preds_probs

    return wrapper


def get_gpc_model(length_scale, n_jobs, seed):

    rbf = ConstantKernel(1.0) * RBF(length_scale=length_scale)
    gpc = GaussianProcessClassifier(rbf, random_state=seed, n_jobs=n_jobs)

    preprocessor = Pipeline(steps=[("scaler", StandardScaler())])

    model = Pipeline(
        steps=[("preprocessor", preprocessor), ("gaussianprocessclassifier", gpc)]
    )

    model.predict = predict_with_confidence(model.predict_proba)

    return model
