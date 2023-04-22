from gp_classifier.data.half_moon import get_half_moon_data
from gp_classifier.model import get_gpc_model
from gp_classifier.utils import eval_metrics


def test_overfit_data(
    data_path="data/historical_data.csv",
    data_repo=None,
    data_version="v2",
    seed=45,
):
    """Test to see whether model can overfit on a couple of examples."""
    X_test, y_test = get_half_moon_data(
        data_path=data_path,
        data_repo=data_repo,
        data_version=data_version,
        split="test",
        test_size=0.02,
        seed=45,
    )
    model = get_gpc_model(length_scale=1.0, n_jobs=2, seed=seed)
    model.fit(X_test, y_test)

    test_preds, _ = model.predict(X_test)
    metrics = eval_metrics(y_test, test_preds)

    assert metrics.get("acc") == 1.0
    assert metrics.get("f1") == 1.0
    assert metrics.get("precision") == 1.0
    assert metrics.get("recall") == 1.0
