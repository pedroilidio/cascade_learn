from numbers import Integral, Real
import joblib
import numpy as np
from sklearn.base import (
    TransformerMixin,
    BaseEstimator,
    MetaEstimatorMixin,
    clone,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import check_scoring
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import HasMethods, Interval
from sklearn.utils.metaestimators import available_if
from sklearn.pipeline import (
    _fit_transform_one,
    _final_estimator_has,
    _print_elapsed_time,
)
from imblearn.pipeline import Pipeline, _fit_resample_one


class Cascade(Pipeline):
    _parameter_constraints = {
        "level": [
            HasMethods(["fit", "transform"]),
            HasMethods(["fit_transform"]),
            HasMethods(["fit_resample"]),
        ],
        "final_estimator": [BaseEstimator, HasMethods(["fit"])],
        "memory": [None, str, HasMethods(["cache"])],
        "verbose": ["boolean"],
        "max_levels": [Interval(Integral, 1, None, closed="left")],
        "scorer": [None, callable, str],  # Can be 'passthrough'
        "stopping_score": [Real],
        "min_improvement": [Interval(Real, 0, None, closed="both")],
        "min_relative_improvement": [Interval(Real, 0, 1, closed="both")],
        "keep_original_features": ["boolean"],
        "validation_size": [
            Interval(Integral, 0, None, closed="left"),
            Interval(Real, 0, 1, closed="neither"),
        ],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        *,
        level,
        final_estimator,
        memory=None,
        verbose=False,
        max_levels=10,
        scorer=None,
        stopping_score=0.0,
        min_improvement=np.inf,
        min_relative_improvement=1.0,
        keep_original_features=True,
        validation_size=0.2,
        random_state=None,
    ):
        self.final_estimator = final_estimator
        self.level = level
        self.memory = memory
        self.verbose = verbose
        self.max_levels = max_levels
        self.scorer = scorer
        self.stopping_score = stopping_score
        self.min_improvement = min_improvement
        self.min_relative_improvement = min_relative_improvement
        self.keep_original_features = keep_original_features
        self.validation_size = validation_size  # TODO: use validation set
        self.random_state = random_state

    @staticmethod
    def _combine_features(X1, X2):
        return np.hstack((X1, X2))

    def fit(self, X, y=None, **fit_params):
        self._validate_steps()
        super().fit(X, y, **fit_params)

    def _validate_steps(self):
        # self._validate_params()  # Done in imblearn.pipeline.Pipeline.fit()
        self.steps = [
            ("final_estimator", clone(self.final_estimator)),
        ]

    def _validate_scorer(self):
        if self.scorer is not None:
            self.scorer_ = check_scoring(
                self.final_estimator,
                scoring=self.scorer,
            )

    def _stop_criterion(self, X, X_val, y, y_val):
        if self.scorer is None:
            return False

        self._final_estimator.fit(X, y)
        X_val = self._apply_transformers(X_val)
        score = self.scorer_(self._final_estimator, X_val, y_val)

        sign = self.scorer_._sign
        stop = sign * (self.stopping_score - score) > 0

        # If this is the first score, we can't compare it to the previous one
        if not hasattr(self, "_last_score"):
            self._last_score = score
            print(f"First score: {score:.4f}")
            return stop

        delta = sign * (self._last_score - score)
        relative_delta = delta / self._last_score

        stop = (
            stop
            or delta < self.min_improvement
            or relative_delta < self.min_relative_improvement
        )
        print(
            f"{score=:.4f} {self._last_score=:.4f} {delta=:.4f}"
            f" {relative_delta=:.4f}"
        )

        self._last_score = score
        return stop

    def _fit(self, X, y=None, **fit_params):
        self._validate_steps()
        self._validate_scorer()

        # Setup the memory
        if self.memory is None or isinstance(self.memory, str):
            memory = joblib.Memory(location=self.memory, verbose=0)
        else:
            memory = self.memory

        fit_transform_one_cached = memory.cache(_fit_transform_one)
        fit_resample_one_cached = memory.cache(_fit_resample_one)

        if self.scorer is not None:
            self.random_state_ = check_random_state(self.random_state)
            X, X_val, y, y_val = train_test_split(
                X,
                y,
                test_size=self.validation_size,
                random_state=self.random_state_,
            )
        else:
            X_val, y_val = None, None

        original_X = X

        for level_count in range(self.max_levels):
            if self._stop_criterion(X, X_val, y, y_val):
                break

            cloned_transformer = clone(self.level)

            # Fit or load from cache the current transformer
            if hasattr(cloned_transformer, "transform") or hasattr(
                cloned_transformer, "fit_transform"
            ):
                new_X, fitted_transformer = fit_transform_one_cached(
                    cloned_transformer,
                    X,
                    y,
                    None,
                    message_clsname=self.__class__.__name__,
                    message=self._log_message(level_count),
                    **fit_params.get("level", {}),
                )
            elif hasattr(cloned_transformer, "fit_resample"):
                new_X, y, fitted_transformer = fit_resample_one_cached(
                    cloned_transformer,
                    X,
                    y,
                    message_clsname=self.__class__.__name__,
                    message=self._log_message(level_count),
                    **fit_params.get("level", {}),
                )
            else:
                raise RuntimeError

            if self.keep_original_features:
                X = self._combine_features(original_X, new_X)
            else:
                X = new_X

            self.steps.insert(
                -1,
                (f"level{level_count}", fitted_transformer),
            )

        return X, y

    def _apply_transformers(self, X):
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = self._combine_features(X, transform.transform(Xt))
        return Xt

    @available_if(_final_estimator_has("predict"))
    def predict(self, X, **predict_params):
        """Transform the data, and apply `predict` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls `predict`
        method. Only valid if the final estimator implements `predict`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **predict_params : dict of string -> object
            Parameters to the ``predict`` called at the end of all
            transformations in the pipeline. Note that while this may be
            used to return uncertainties from some models with return_std
            or return_cov, uncertainties that are generated by the
            transformations in the pipeline are not propagated to the
            final estimator.

            .. versionadded:: 0.20

        Returns
        -------
        y_pred : ndarray
            Result of calling `predict` on the final estimator.
        """
        Xt = self._apply_transformers(X)
        return self.steps[-1][1].predict(Xt, **predict_params)

    @available_if(_final_estimator_has("predict_proba"))
    def predict_proba(self, X, **predict_proba_params):
        """Transform the data, and apply `predict_proba` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `predict_proba` method. Only valid if the final estimator implements
        `predict_proba`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **predict_proba_params : dict of string -> object
            Parameters to the `predict_proba` called at the end of all
            transformations in the pipeline.

        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Result of calling `predict_proba` on the final estimator.
        """
        Xt = self._apply_transformers(X)
        return self.steps[-1][1].predict_proba(Xt, **predict_proba_params)

    @available_if(_final_estimator_has("decision_function"))
    def decision_function(self, X):
        """Transform the data, and apply `decision_function` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `decision_function` method. Only valid if the final estimator
        implements `decision_function`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_score : ndarray of shape (n_samples, n_classes)
            Result of calling `decision_function` on the final estimator.
        """
        Xt = self._apply_transformers(X)
        return self.steps[-1][1].decision_function(Xt)

    @available_if(_final_estimator_has("score_samples"))
    def score_samples(self, X):
        """Transform the data, and apply `score_samples` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `score_samples` method. Only valid if the final estimator implements
        `score_samples`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_score : ndarray of shape (n_samples,)
            Result of calling `score_samples` on the final estimator.
        """
        Xt = self._apply_transformers(X)
        return self.steps[-1][1].score_samples(Xt)

    @available_if(_final_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X, **predict_log_proba_params):
        """Transform the data, and apply `predict_log_proba` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `predict_log_proba` method. Only valid if the final estimator
        implements `predict_log_proba`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **predict_log_proba_params : dict of string -> object
            Parameters to the ``predict_log_proba`` called at the end of all
            transformations in the pipeline.

        Returns
        -------
        y_log_proba : ndarray of shape (n_samples, n_classes)
            Result of calling `predict_log_proba` on the final estimator.
        """
        Xt = self._apply_transformers(X)
        return self.steps[-1][1].predict_log_proba(Xt, **predict_log_proba_params)

    def _can_transform(self):
        return self._final_estimator == "passthrough" or hasattr(
            self._final_estimator, "transform"
        )

    @available_if(_can_transform)
    def transform(self, X):
        """Transform the data, and apply `transform` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `transform` method. Only valid if the final estimator
        implements `transform`.

        This also works where final estimator is `None` in which case all prior
        transformations are applied.

        Parameters
        ----------
        X : iterable
            Data to transform. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_transformed_features)
            Transformed data.
        """
        Xt = self._apply_transformers(X)
        return self.steps[-1][1].transform(Xt)

    def _can_inverse_transform(self):
        return False

    @available_if(_final_estimator_has("score"))
    def score(self, X, y=None, sample_weight=None):
        """Transform the data, and apply `score` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `score` method. Only valid if the final estimator implements `score`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        y : iterable, default=None
            Targets used for scoring. Must fulfill label requirements for all
            steps of the pipeline.

        sample_weight : array-like, default=None
            If not None, this argument is passed as ``sample_weight`` keyword
            argument to the ``score`` method of the final estimator.

        Returns
        -------
        score : float
            Result of calling `score` on the final estimator.
        """
        Xt = self._apply_transformers(X)
        score_params = {}
        if sample_weight is not None:
            score_params["sample_weight"] = sample_weight
        return self.steps[-1][1].score(Xt, y, **score_params)
