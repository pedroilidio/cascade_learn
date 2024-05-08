from sklearn.base import MetaEstimatorMixin, clone, is_classifier
from imblearn.base import BaseSampler


class WeakLabelImputer(BaseSampler, MetaEstimatorMixin):
    def __init__(self, estimator, threshold=0.8):
        self.estimator = estimator
        self.threshold = threshold

    def _fit_resample(self, X, y):
        if not is_classifier(self.estimator):
            raise ValueError(
                "'estimator' parameter must be a classifier instance. "
                "Got {self.estimator}.",
            )

        classifier = clone(self.estimator).fit(X, y)

        y_pred = classifier.predict(X)
        proba = classifier.predict_proba(X)

        # Recover true labels for samples with low confidence
        mask = proba.max(axis=-1) < self.threshold
        y_pred[mask] = y[mask]

        return X, y_pred
