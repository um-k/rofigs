import os
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils.multiclass import check_classification_targets
import scipy.sparse
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error as mse
from src.constants import seed_everything


seed_everything()


def load_data(dataset, fold=0):
    """ Load preprocessed data, split into 3 subsets: train, validation, and test. 
        Data needs to be saved in data/dataset folder.
    Args:
        dataset (str): name of the dataset
        fold (int): fold number
    """
    
    ds_dir = os.path.join(os.path.abspath('..'), "data", dataset)
    # ds_dir = os.path.join(os.getcwd(), "data", dataset)
    if not os.path.exists(ds_dir):
        print(f"Folder {ds_dir} does not exist.")
        return
    elif not os.listdir(ds_dir):
        print(f"Folder {ds_dir} is empty.")
        return
    
    else:
        data = dict()
        loaded_train = np.load(os.path.join(ds_dir, f"train_fold{fold}.npz"), allow_pickle=True)
        data["train"] = (loaded_train["X"], loaded_train["y"])
        loaded_val = np.load(os.path.join(ds_dir, f"val_fold{fold}.npz"), allow_pickle=True)
        data["val"] = (loaded_val["X"], loaded_val["y"])
        loaded_test = np.load(os.path.join(ds_dir, f"test_fold{fold}.npz"), allow_pickle=True)
        data["test"] = (loaded_test["X"], loaded_test["y"])
        return (data["train"], data["val"], data["test"])

    
def load_final_data(dataset, fold=0):
    """ Load preprocessed data, split into 2 subsets: train+validation and test 
        Data needs to be saved in data/dataset folder.
    Args:
        dataset (str): name of the dataset
        fold (int): fold number
    """
    
    ds_dir = os.path.join(os.path.abspath('..'), "data", dataset)
    # ds_dir = os.path.join(os.getcwd(), "data", dataset)
    if not os.path.exists(ds_dir):
        print(f"Folder {ds_dir} does not exist.")
        return
    elif not os.listdir(ds_dir):
        print(f"Folder {ds_dir} is empty.")
        return
    
    else:
        data = dict()
        loaded_train = np.load(os.path.join(ds_dir, f"train_final_fold{fold}.npz"), allow_pickle=True)
        data["train_final"] = (loaded_train["X"], loaded_train["y"])
        loaded_test = np.load(os.path.join(ds_dir, f"test_final_fold{fold}.npz"), allow_pickle=True)
        data["test_final"] = (loaded_test["X"], loaded_test["y"])
        return (data['train_final'], data['test_final'])
    
    
def print_split(node, mode):
    """Print the split information of a node in RO-FIGS model
    Args:
        node (Node): a node in the tree
        mode (str): either "root" or "split"
    """
    out = ""
    
    for k in range(len(node.features)):
        if abs(node.weights[k]) > 0.0:
            if node.weights[k] < 0.0:
                out += f" - {abs(node.weights[k]):0.3f} * X_{node.features[k]}"
            else:
                out += f" + {node.weights[k]:0.3f} * X_{node.features[k]}"
    if mode=="root":
        out += f" <= {node.threshold:0.3f} (Tree #{node.tree_num} root)"
    elif mode=="split":
        out += f" <= {node.threshold:0.3f} (split)"
                                                                                
    # modify the beginning of the output 
    if out[1] == "+":
        out = out[3:]
    elif out[1] == "-":
        out = out[1:]
    return out


def extract_info(clf_regr, X, y):
    """Return FIGS stump-like information from the odt model (i.e., oblique stump)
    Args:
        clf_regr (odt): fitted odt model (see odt.py for details)
        X (np.array): input data
        y (np.array): target data
    """
    
    info_dict = dict()    

    left_side = np.dot(X[:,clf_regr.odt_info["features"]], clf_regr.odt_info["weights"])
    eq = left_side <= clf_regr.odt_info["threshold"]    # eg (198,)  ... an array with True and False values
    lefties=np.argwhere(eq==True).reshape(-1)           # eg (44,)
    righties=np.argwhere(eq==False).reshape(-1)         # eg (154,)
    
    info_dict["features"] = clf_regr.odt_info["features"]
    info_dict["weights"] = clf_regr.odt_info["weights"]
    info_dict["threshold"] = clf_regr.odt_info["threshold"]
    
    n_node_samples = np.array([len(lefties)+len(righties), len(lefties), len(righties)])
    info_dict["n_node_samples"] = n_node_samples
    
    if len(lefties)==0 or len(righties)==0:
        raise ZeroDivisionError

    lefties_value = y[lefties].mean()
    righties_value = y[righties].mean()
    
    predictions_left = np.full((len(lefties), 1), lefties_value)
    predictions_right = np.full((len(righties), 1), righties_value)

    left_mse = mse(y[lefties], predictions_left)        
    right_mse = mse(y[righties], predictions_right) 
    
    left_value = np.array([lefties_value])
    right_value = np.array([righties_value])
    
    dummy = DummyRegressor() 
    dummy.fit(X, y)
    split_mse = mse(y, dummy.predict(X))
    split_value = dummy.predict([X[0,:]])   # np.mean(y)

    impurity = np.array([split_mse, left_mse, right_mse])
    info_dict["impurity"] = impurity
    value = np.array([[split_value], [left_value], [right_value]])
    info_dict["value"] = value

    return info_dict


def leaf_impurity(X, y):
    """Return the impurity of a leaf node """
    dummy = DummyRegressor() 
    dummy.fit(X, y)
    mse_imp = mse(y, dummy.predict(X))
    return mse_imp


# From imodels.util.arguments import check_fit_arguments
def check_fit_arguments(model, X, y, feature_names):
    """Process arguments for fit and predict methods """
    if isinstance(model, ClassifierMixin):
        model.classes_, y = np.unique(y, return_inverse=True)  # deals with str inputs
        check_classification_targets(y)

    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            model.feature_names_ = X.columns
        elif isinstance(X, list):
            model.feature_names_ = ['X' + str(i) for i in range(len(X[0]))]
        else:
            model.feature_names_ = ['X' + str(i) for i in range(X.shape[1])]
    else:
        model.feature_names_ = feature_names
    if scipy.sparse.issparse(X):
        X = X.toarray()
    X, y = check_X_y(X, y)
    _, model.n_features_in_ = X.shape
    assert len(model.feature_names_) == model.n_features_in_, 'feature_names should be same size as X.shape[1]'
    y = y.astype(float)
    return X, y, model.feature_names_

