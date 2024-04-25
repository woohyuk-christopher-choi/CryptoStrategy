import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import optuna as opt
import dictionary.fund_scoring_2nd_dict as config
from optuna import Trial, visualization
from optuna.samplers import TPESampler
import plotly


class OptunaObject(object):
    '''
    학습데이터셋으로 구성된 array와 학습팧 모델의 이름을 받아 최적화할 파라미터 공간과 스코어를 정의

    Args:
        x,y (DataFrame): 학습할 데이터
        model_name (str): 학습할 모형의 이름

    Returns:
        파라미터 최적화시 최적화할 스코어 함수
    '''

    def __init__(self, X_train: pd.DataFrame, Y_train: pd.Series, X_val: pd.DataFrame, Y_val: pd.Series, model_name: str):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.model_name = model_name

    def __call__(self, trial):

        if self.model_name == 'ElasticNet':

            params = {
                        'alpha': trial.suggest_float('alpha', 0.0001, 0.1),
                        'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 0.9, step=0.2)
                    }
            model = ElasticNet(**params)

        elif self.model_name == 'Lasso':
            params = {
                        'alpha': trial.suggest_float('alpha', 0, 0.1)
                    }
            model = Lasso(**params)

        elif self.model_name == 'LGBMRegressor':
            params = {
                'objective': trial.suggest_categorical('objective', ['l2','l1','regression']),
                'metric': 'l2',
                'verbose': -1,
                'njobs': 5,
                'num_iterations':trial.suggest_int('num_iterations', 10, 100),
                'learning_rate': 0.01,
                'max_depth': trial.suggest_int('max_depth',3 , 12),
                'bosting_type': trial.suggest_categorical('bosting_type', ['gbdt', 'dart']),
                'lambda_l1' : trial.suggest_float('lambda_l1',1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                'num_leaves': trial.suggest_int('num_leaves',2,256),
                'feature_fraction':trial.suggest_float('feature_fraction',0.4,1.0),
                'bagging_fraction':trial.suggest_float('bagging_fraction',0.4,1.0),
                'bagging_freq':trial.suggest_int('bagging_freq',1,7),
                'min_child_samples':trial.suggest_int('min_child_samples',5,100)
            }
            model = lgb.LGBMRegressor(**params)

        elif self.model_name == 'RandomForestRegressor':
            params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 200, step=50),
                        'max_depth': trial.suggest_int('max_depth', 3, 20, step=1),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
                        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
                    }
            model = RandomForestRegressor(**params)

        elif self.model_name == 'SVR':
            params = {
                        'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
                        'C': trial.suggest_float('C', 0.1, 100, log=True),
                        'epsilon': trial.suggest_float('epsilon', 0.01 ,1, log=True)
                    }
            model = SVR(**params)

        model.fit(self.X_train, self.Y_train)
        Y_pred = model.predict(self.X_val)

        mse = mean_squared_error(self.Y_val, Y_pred)

        return mse


def find_params(X_train: pd.DataFrame, Y_train: pd.DataFrame, X_val: pd.DataFrame, Y_val: pd.DataFrame, model_name='ElasticNet', n_trials = 100):


    timeout = 1000
    objective = OptunaObject(X_train, Y_train, X_val, Y_val, model_name)
    study = opt.create_study(pruner=opt.pruners.HyperbandPruner(), direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    return study.best_params, study.best_trial

















































