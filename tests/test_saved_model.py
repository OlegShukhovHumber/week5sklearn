import joblib
import numpy as np


def test_saved_model_predicts_the_same_after_reloading(
    fitted_pipeline, split, tmp_path
):
    _, X_test, _, _ = split
    path = tmp_path / "pipeline.joblib"
    joblib.dump(fitted_pipeline, path)
    reloaded = joblib.load(path)
    assert np.allclose(
        fitted_pipeline.predict_proba(X_test), reloaded.predict_proba(X_test)
    )
