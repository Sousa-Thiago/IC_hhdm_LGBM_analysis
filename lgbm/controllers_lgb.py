import numpy as np
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold


class LGBLearner:
    """
    Class for training LGB model
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
        clf = lgb.LGBMClassifier(
            objective="binary",
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
        Train LGB model
        """
        if self.features is None:
            raise ValueError("Features is not set. Impossible to train.")

#         dtrain = None
        if missing_values is np.nan:
            dtrain = lgb.Dataset(
                data=self.X,
                label=self.Y,
                weight=self.W,
#                 missing=missing_values,
#                 feature_names=self.features
            )
        else:
            raise ValueError("Condição não atendida para dtrain")

        if dtrain is not None:
            self.clf = lgb.train(
                params=hyperparameters, train_set=dtrain, num_boost_round=num_boost_round
            )


    def trainV2(self, hyperparameters, use_label_encoder=False):
        """
        Train LGB model
        """
        model = lgb.LGBMClassifier(
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


class LGBModel:
    """
    Load LGB model
    """

    def __init__(self, classifier=None, model_fpath=None):
        if classifier and model_fpath:
            raise ValueError("Specify a classifier or model_fpath not both.")
        if model_fpath:
            classifier = lgb.Booster()
            classifier.load_model(model_fpath)

        self.clf = classifier

    def predict(self, X, features, missing_values=np.nan):
        """
        Predict values
        """
        lgb_test = lgb.DataSet(
            data=X,
            missing=missing_values,
            feature_names=features,
        )
        return self.clf.predict(lgb_test)
