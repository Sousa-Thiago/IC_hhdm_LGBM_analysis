import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold


class XGBLearner:
    """
    Class for training XGB model
    """

    def __init__(self, X, Y, W, features=None, random_state=42, njobs=-1):
        self.X = X
        self.Y = Y
        self.W = W
        self.features = features
        self.random_state = random_state
        self.njobs = njobs
        self.clf = None

    def find_hyperparams(
        self,
        hyperparams_grid,
        n_splits,
        n_iter,
        scoring="f1",
        use_label_encoder=False,
        verbose=3,
    ):
        """
        Find best hyperparameters using KFold CrossValidation + RandomizedSearchCV
        """
        clf = xgb.XGBClassifier(
            objective="binary:logistic",
            nthread=self.njobs,
            random_state=self.random_state,
            use_label_encoder=use_label_encoder,
        )
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        rands = RandomizedSearchCV(
            clf,
            param_distributions=hyperparams_grid,
            n_iter=n_iter,
            scoring=scoring,
            n_jobs=self.njobs,
            cv=skf.split(self.X, self.Y),
            verbose=verbose,
            random_state=self.random_state,
        )
        rands.fit(self.X, self.Y, sample_weight=self.W)
        return {
            "hyperparameters": rands.best_params_,
            "classifier": rands.best_estimator_,
        }

    def train(self, hyperparameters, num_boost_round, missing_values=np.nan):
        """
        Train XGB model
        """
        if self.features is None:
            raise ValueError("Features is not set. Impossible to train.")

        dtrain = xgb.DMatrix(
            data=self.X,
            label=self.Y,
            weight=self.W,
            missing=missing_values,
            feature_names=self.features,
        )
        self.clf = xgb.train(
            params=hyperparameters, dtrain=dtrain, num_boost_round=num_boost_round
        )

    def trainV2(self, hyperparameters, use_label_encoder=False):
        """
        Train XGB model
        """
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            nthread=self.njobs,
            random_state=self.random_state,
            use_label_encoder=use_label_encoder,
            **hyperparameters
        )
        self.clf = model.fit(self.X, self.Y, sample_weight=self.W)

    def save_model(self, model_fpath):
        """
        Save model
        """
        self.clf.save_model(model_fpath)

    def dump_model(self, model_fpath):
        """
        Dump raw model and featmap
        """
        self.clf.dump_model(model_fpath)


class XGBModel:
    """
    Load XGB model
    """

    def __init__(self, classifier=None, model_fpath=None):
        if classifier and model_fpath:
            raise ValueError("Specify a classifier or model_fpath not both.")
        if model_fpath:
            classifier = xgb.Booster()
            classifier.load_model(model_fpath)

        self.clf = classifier

    def predict(self, X, features, missing_values=np.nan):
        """
        Predict values
        """
        xgb_test = xgb.DMatrix(
            data=X,
            missing=missing_values,
            feature_names=features,
        )
        return self.clf.predict(xgb_test)
