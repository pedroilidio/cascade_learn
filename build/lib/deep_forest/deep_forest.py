import joblib
from sklearn.datasets import load_iris
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from cascade_forests.tree_embedder import ClassifierTransformer
from cascade import Cascade

random_forest_transformer = ClassifierTransformer(
    CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=500, max_features="sqrt"),
        cv=3,
        ensemble=True,
    )
)
extra_trees_transformer = ClassifierTransformer(
    CalibratedClassifierCV(
        ExtraTreesClassifier(n_estimators=500, max_features=1),
        cv=3,
        ensemble=True,
    )
)

level_estimator = FeatureUnion(
    [
        ("xt", extra_trees_transformer),
        ("rf", random_forest_transformer),
    ]
)

cascade_forest = Cascade(
    level=level_estimator,
    final_estimator=RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
    ),
    scorer="neg_mean_squared_error",
    stopping_score=-0.0001,
    # min_improvement=0.00001,
    max_levels=10,
    verbose=True,
    random_state=10,
)

if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    cascade_forest.fit(X, y)
    joblib.dump(cascade_forest, "cascade_forest.joblib")
