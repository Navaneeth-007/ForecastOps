import numpy as np
import pandas as pd
import pytest

from app.ingestion.data_loader import _generate_synthetic, validate_schema
from app.serving.recursive import recursive_predict


class LagModel:
    def predict(self, frame):
        return frame["lag_1h"].to_numpy() + 1


def test_recursive_forecast_uses_previous_prediction():
    history = _generate_synthetic(10)
    dates = pd.date_range(history.date.max() + pd.Timedelta(hours=1), periods=3, freq="h")
    actual = recursive_predict(LagModel(), history, dates)
    np.testing.assert_allclose(actual, history.demand_kwh.iloc[-1] + np.arange(1, 4))


@pytest.mark.parametrize("kind", ["duplicate", "gap", "negative", "infinite", "null"])
def test_reject_invalid_history(kind):
    data = _generate_synthetic(10)
    if kind == "duplicate":
        data.loc[1, "date"] = data.loc[0, "date"]
    elif kind == "gap":
        data = data.drop(index=10)
    elif kind == "negative":
        data.loc[0, "demand_kwh"] = -1
    elif kind == "infinite":
        data.loc[0, "demand_kwh"] = np.inf
    else:
        data.loc[0, "date"] = pd.NaT
    with pytest.raises(ValueError):
        validate_schema(data)


def test_schema_does_not_mutate_input():
    data = _generate_synthetic(10)
    original = data.copy(deep=True)
    validate_schema(data)
    pd.testing.assert_frame_equal(original, data)
