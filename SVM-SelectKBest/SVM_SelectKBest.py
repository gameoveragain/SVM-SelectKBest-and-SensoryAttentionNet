import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import f_classif
from sklearn.svm import SVC

class SVM_SelectKBest(BaseEstimator, TransformerMixin):
    """Feature selector that uses simulated annealing to optimize the feature subset for an SVM classifier."""
    
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.best_k = None
        self.best_score = -np.inf
        self.indices_ = None
        self.scores_ = None
        self.best_model_ = None  # Stores the best model

    def fit(self, X_train, Y_train, X_val, Y_val):
        """Fit the selector based on input data using simulated annealing."""
        # Calculate ANOVA F-value for the features and cache their indices based on the scores
        self.scores_, _ = f_classif(X_train, Y_train)
        self.indices_ = np.argsort(self.scores_)[::-1]
        
        # Parameters for the SVM grid search
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100, 1000, 10000],
            'kernel': ['rbf', 'linear', 'sigmoid', 'poly'],
            'class_weight': [None, 'balanced']
        }

        # Initial settings for simulated annealing
        initial_temperature = 1000
        min_temperature = 1
        cooling_rate = 0.95
        current_temperature = initial_temperature
        n_features = X_train.shape[1]
        k = 1  # Initial number of features

        # Simulated annealing process
        while current_temperature > min_temperature:
            step = np.random.randint(-10, 30)  # Random step for feature number change
            new_k = max(1, min(n_features, k + step))  # New candidate number of features

            # Select features and subset the training and validation data
            selected_features = self.indices_[:new_k]
            X_train_selected = X_train[:, selected_features]
            X_val_selected = X_val[:, selected_features]

            # Train SVM and evaluate performance
            svm = SVC(gamma='scale', random_state=self.random_state)
            grid_search = GridSearchCV(svm, param_grid, cv=5, refit=True)
            grid_search.fit(X_train_selected, Y_train)
            performance_score = 0.5 * (grid_search.score(X_val_selected, Y_val) +
                                       grid_search.score(X_train_selected, Y_train))

            # Metropolis criterion to decide whether to update the best model
            if performance_score > self.best_score:
                self.best_score = performance_score
                self.best_k = new_k
                self.best_model_ = grid_search  # Save the best model found so far
                k = new_k
            elif performance_score == self.best_score and new_k < self.best_k:
                self.best_k = new_k
                self.best_model_ = grid_search  # Save the best model found so far
                k = new_k    

            current_temperature *= cooling_rate

        return self

    def transform(self, X):
        """Transform the dataset by selecting only the best features determined by fit."""
        if self.best_k is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        selected_features = self.indices_[:self.best_k]
        return X[:, selected_features]

    def predict(self, X):
        """Predict the target using the best model found."""
        if self.best_model_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        X_transformed = self.transform(X)
        return self.best_model_.predict(X_transformed)

    def fit_transform(self, X_train, Y_train, X_val, Y_val):
        """Fit the model and transform the training set simultaneously."""
        self.fit(X_train, Y_train, X_val, Y_val)
        return self.transform(X_train)