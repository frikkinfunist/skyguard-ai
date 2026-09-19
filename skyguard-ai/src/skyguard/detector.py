"""Layer 3: multivariate consistency scoring across sensors (Isolation Forest)."""

import pandas as pd
from sklearn.ensemble import IsolationForest


def multivariate_scores(resid: pd.DataFrame, contamination: float = 0.02, seed: int = 42):
    """Fit Isolation Forest on seasonal residuals; return (scores, flag).

    scores: decision_function output, higher = more normal.
    flag:   True where the model calls the point an outlier.
    """
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)
    valid = resid.dropna()
    model.fit(valid)

    scores = pd.Series(model.decision_function(valid), index=valid.index)
    flag = pd.Series(model.predict(valid) == -1, index=valid.index)
    return scores, flag
