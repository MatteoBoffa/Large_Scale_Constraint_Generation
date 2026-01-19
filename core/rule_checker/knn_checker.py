from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from core.rule_checker.shallow_classifier import Shallow_Classifier


class KNNChecker(Shallow_Classifier):
    """Rule compliance checker using tuned KNN classification."""

    def tune_hyperparameters(self, X_train, y_train, cv=3, seed=42, n_proc=1):
        """
        Perform grid search to find optimal hyperparameters.

        Args:
            X_train: Training features
            y_train: Training labels
            cv: Number of cross-validation folds
            n_proc: Number of processes
        """
        param_grid = {
            # Keep neighborhood small to limit computation
            "n_neighbors": [3, 5, 7, 9, 11],
            # Common and well-motivated options
            "weights": ["uniform", "distance"],
            # Cosine often works better for embeddings
            "metric": ["euclidean", "cosine"],
        }

        base_knn = KNeighborsClassifier()

        grid_search = GridSearchCV(
            estimator=base_knn,
            param_grid=param_grid,
            cv=cv,
            scoring="f1",
            n_jobs=n_proc,
            verbose=2,
        )

        grid_search.fit(X_train, y_train)

        self.best_params_ = grid_search.best_params_
        self.classifier = grid_search.best_estimator_

        return grid_search.best_params_, grid_search.cv_results_
