from sklearn.metrics import mean_squared_error, make_scorer
from typing import Callable
from hyperopt import hp


def load_validation_scorer_fn() -> (str, Callable):

    return ("mean_squared_error", make_scorer(mean_squared_error, greater_is_better = False))

def load_search_space_fn() -> Callable:

    def search_space() -> dict:
        return {
            "fit_intercept" : hp.choice("lr_fit_intercept", [True, False]),
            "positive" : hp.choice("lr_positive", [True, False]),
        }
    
    return search_space