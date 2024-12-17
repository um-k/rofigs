import numpy as np
import random

RANDOM_STATE = 12345 

def seed_everything(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    
    
datasets = ["blood", "diabetes", "breast-w", "ilpd", "monks2", "climate", "kc2", "pc1", "kc1", "heart", "tictactoe", "wdbc", 
            "churn", "pc3", "biodeg", "credit", "spambase", "credit-g", "friedman", "usps", "bioresponse", "speeddating"]


# dataset: (min_imp_dec, beam_size)
opt_config = {'blood': (5, 2),
              'diabetes': (10, 8),
              'breast-w': (1, 9),
              'ilpd': (0.1, 5),
              'monks2': (0.1, 17),
              'climate': (0.1, 18),
              'kc2': (10, 21),
              'pc1': (0.1, 4),
              'kc1': (0.05, 4),
              'heart': (10, 26),
              'tictactoe': (0.05, 13),
              'wdbc': (1, 30),
              'churn': (5, 33),
              'pc3': (0.1, 18),
              'biodeg': (10, 41),
              'credit': (1, 12),
              'spambase': (0.1, 57),
              'credit-g': (1, 2),
              'friedman': (10, 25),
              'usps': (1, 256),
              'bioresponse': (0.05, 419),
              'speeddating': (10, 125)
}

includes_categorical = ["ilpd", "monks2", "heart", "tictactoe", "churn", "credit", "credit-g", "speeddating"] 

baselines_ohe = ["mlp", "oblens", "xgboost", "dt", "rf", "figs", "etc", "odt", "mt"] 
