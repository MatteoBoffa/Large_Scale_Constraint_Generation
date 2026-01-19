from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from core.rule_checker.shallow_classifier import Shallow_Classifier


class RandomForestChecker(Shallow_Classifier):
    """Rule compliance checker using tuned Random Forest classification."""

    def tune_hyperparameters(self, X_train, y_train, cv=3, seed=42, n_proc=1):
        """Perform randomized search to find optimal hyperparameters for Random Forest."""

        param_grid = {
            "n_estimators": [100, 200, 400],
            "max_depth": [10, 20],
            "max_features": ["sqrt"],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 3, 5],
            "bootstrap": [True, False],
            "class_weight": [None],
        }

        base_rf = RandomForestClassifier(random_state=seed, n_jobs=n_proc)

        grid_search = GridSearchCV(
            estimator=base_rf,
            param_grid=param_grid,
            cv=cv,
            scoring="f1",
            n_jobs=1,
            verbose=2,
        )

        grid_search.fit(X_train, y_train)
        self.best_params_ = grid_search.best_params_
        self.classifier = grid_search.best_estimator_

        return grid_search.best_params_, grid_search.cv_results_
