import numpy as np
import scipy.sparse as sp
from spyct.node import Node, traverse_tree
from spyct.grad_split import GradSplitter
from spyct.data import data_from_np_or_sp, relative_impurity
from spyct._matrix import memview_to_SMatrix, memview_to_DMatrix, csr_to_SMatrix
from joblib import Parallel, delayed

from src.constants import seed_everything
seed_everything()

DTYPE = 'f'


# Based on spyct.model
class Model:

    def __init__(self,
                 splitter='grad', objective='mse', boosting=False, boosting_step=0.1, num_trees=1, max_features=1, bootstrapping=None, 
                 max_depth=1, min_examples_to_split=5, min_impurity_decrease=0.05, n_jobs=1, standardize_descriptive=True, 
                 standardize_clustering=True, max_iter=100, lr=0.1, C=10, tol=1e-2, eps=1e-8, adam_beta1=0.9, adam_beta2=0.999, 
                 random_state=None, splitting_features=None):

        self.splitting_features = splitting_features
        self.num_trees = num_trees
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_examples_to_split = min_examples_to_split
        self.n_jobs = n_jobs
        self.min_impurity_decrease = min_impurity_decrease

        # universal parameters
        self.splitter = splitter
        self.objective = objective
        self.boosting = boosting
        self.boosting_step = boosting_step
        self.max_iter = max_iter
        self.standardize_descriptive = standardize_descriptive
        self.standardize_clustering = standardize_clustering
        self.tol = tol
        self.eps = eps
        self.C = C
        
        # variance parameters
        self.lr = lr
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2

        if type(random_state) is int: self.rng = np.random.RandomState(random_state)
        elif type(random_state) is np.random.RandomState: self.rng = random_state
        else: self.rng = np.random.RandomState()

        if bootstrapping is not None: self.bootstrapping = bootstrapping
        else: self.bootstrapping = num_trees > 1

        if splitter not in ['svm', 'grad']: raise ValueError('Unknown splitter specified. Supported values are "grad" and "svm".')
        if objective not in ['mse', 'dot']: raise ValueError('Unknown objective function specified. Possible values are "mse" and "dot".')

        self.trees = None                   # after fitting the model, this holds the list of trees in the ensemble
        self.sparse_target = None           # bool denoting if the matrix of target values is sparse
        self.num_targets = None             # the number of target variables
        self.num_nodes = None               # the number of nodes in the ensemble
        self.feature_importances = None     # the importance of each feature based on the learned ensemble
        self.max_relative_impurity = None   # the maximum relative impurity remained after a split

        
    def fit(self, descriptive_data, target_data, clustering_data=None, clustering_weights=None):
        # descriptive_data (shape = [n_samples, n_features]) - features of the training examples (used for splitting the examples)
        # target_data (shape = [n_samples, n_outputs]) - target variables of the training examples (what the model will predict)
        # optional:
        #       clustering_data (shape = [n_samples, n_clustering_variables]) - data used to evaluate the splits;
        #                       by default = target_data; we optimize the splits according to the variables we wish to predict
        #       param clustering_weights (shape = [n_clustering_variables]) - by default, all targets have the same weight.
        # descriptive data: (n x d)
        # target data:      (n x t)
        
        # data checks
        if len(descriptive_data.shape) != 2: raise ValueError("Descriptive data must have exactly 2 dimensions.")
        if clustering_data is not None and len(clustering_data.shape) != 2: raise ValueError("Clustering data must have exactly 2 dimensions.")
        if len(target_data.shape) != 2: raise ValueError("Target data must have exactly 2 dimensions.")

        # Calculate the number of features to consider at each split
        if type(self.max_features) is int:      num_features = self.max_features
        elif type(self.max_features) is float:  num_features = self.max_features * descriptive_data.shape[1]
        elif self.max_features == 'sqrt':       num_features = np.sqrt(descriptive_data.shape[1])
        elif self.max_features == 'log':        num_features = np.log2(descriptive_data.shape[1])
        else:                                   raise ValueError("The max_features parameter was specified incorrectly.")
        num_features = max(1, int(np.ceil(num_features)))

        # If data is sparse, make sure it has no missing values and the format is CSR. Make sure the numeric precision is correct.
        bias_col = np.ones([descriptive_data.shape[0], 1], dtype=DTYPE)  # column of ones for bias calculation
        if sp.issparse(descriptive_data):
            descriptive_data = descriptive_data.astype(DTYPE, copy=False)
            descriptive_data = sp.hstack((descriptive_data, bias_col)).tocsr()
        else:
            descriptive_data = descriptive_data.astype(DTYPE, order='C', copy=False)
            descriptive_data = np.hstack((descriptive_data, bias_col))
        self.sparse_target = sp.issparse(target_data)
        if self.sparse_target:
            target_data = target_data.astype(DTYPE, copy=False)
            target_data = target_data.tocsr()
        else:
            target_data = target_data.astype(DTYPE, order='C', copy=False)
        clustering_data = target_data       # because clustering_data and clustering_weights are None
       
        # Initialize everything
        all_data = data_from_np_or_sp(descriptive_data, target_data, clustering_data)
        self.num_nodes = 0
        self.num_targets = target_data.shape[1]
        #print(f"### target data of length 1?? -- {target_data.shape[1]} ###")   #CHECK
        self.max_relative_impurity = 1 - self.min_impurity_decrease
        self.feature_importances = np.zeros(descriptive_data.shape[1] - 1)

        # Function that wraps tree building for parallelization. Bootstrapping if more than one tree.
        def tree_builder(seed):
            rng = np.random.RandomState(seed)
            if self.bootstrapping:
                rows = rng.randint(target_data.shape[0], size=target_data.shape[0], dtype=np.intp)
                data = all_data.take_rows(rows)
            else:
                data = all_data
            return self._grow_tree(data, clustering_weights, num_features, rng)
        
        # Learn the trees
        seeds = self.rng.randint(10**9, size=self.num_trees)
        results = [tree_builder(seeds[i]) for i in range(self.num_trees)]
        
        # Collect the results
        self.trees = []
        for node_list, importances, odt_info in results:
                        
            self.trees.append(node_list)
            self.num_nodes += len(node_list)
            self.feature_importances += importances
            if len(node_list) == 1:
                odt_info["one_node_only"] = True
            else:
                odt_info["one_node_only"] = False
            self.odt_info = odt_info

            if "features" not in odt_info:
                odt_info["error"] = True
   

        self.feature_importances /= self.num_trees
        return self


    def predict(self, descriptive_data, used_trees=None):
        # The predictions made by the model -> returns an array of shape = [n_samples, n_outputs]
        # descriptive_data: array-like or sparse matrix of shape = [n_samples, n_features]
        # used_trees=None - all trees used to make a prediction (int for a subset)

        # Add the bias column
        n = descriptive_data.shape[0]
        descriptive_data = descriptive_data.astype(DTYPE, copy=False)
        if sp.issparse(descriptive_data): descriptive_data = csr_to_SMatrix(descriptive_data.to_csr())

        predictions = np.zeros((n, self.num_targets), dtype=DTYPE)
        stack = np.vstack

        def predictor(node_list, descriptive_data):
            descriptive_data = descriptive_data.copy()
            return stack([traverse_tree(node_list, descriptive_data, i) for i in range(n)])

        n_trees = self.num_trees if used_trees is None else used_trees

        for node_list in self.trees[:n_trees]:
            predictions += predictor(node_list, descriptive_data)
            
        if not self.boosting:
            predictions /= n_trees
        return predictions

    def get_params(self, **kwargs):
        return {
            'splitter': self.splitter,
            'objective': self.objective,
            'num_trees': self.num_trees,
            'max_features': self.max_features,
            'bootstrapping': self.bootstrapping,
            'max_depth': self.max_depth,
            'min_examples_to_split': self.min_examples_to_split,
            'min_impurity_decrease': self.min_impurity_decrease,
            'n_jobs': self.n_jobs,
            'standardize_descriptive': self.standardize_descriptive,
            'standardize_clustering': self.standardize_clustering,
            'max_iter': self.max_iter,
            'lr': self.lr,
            'adam_beta1': self.adam_beta1,
            'adam_beta2': self.adam_beta2,
            'tol': self.tol,
            'C': self.C,
            'eps': self.eps,
        }

    def set_params(self, **params):

        for key, value in params.items():
            if key == 'random_state':
                if type(value) is int:
                    self.rng = np.random.RandomState(value)
                elif type(value) is np.random.RandomState:
                    self.rng = value
                else:
                    self.rng = np.random.RandomState()
            else:
                setattr(self, key, value)

        if 'bootstrapping' not in params:
            self.bootstrapping = self.num_trees > 1
        if self.splitter not in ['svm', 'grad']:
            raise ValueError('Unknown splitter specified. Supported values are "grad" and "svm".')
        if self.objective not in ['mse', 'dot']:
            raise ValueError('Unknown objective function specified. Possible values are "mse" and "dot".')

        return self


    ############################## ---------- GROW A SINGLE TREE ---------- ##############################
    
    def _grow_tree(self, root_data, clustering_weights, num_features, rng):

        if self.splitter == 'grad':
            splitter = GradSplitter(n=root_data.n, d=num_features+1, c=root_data.c,
                                    clustering_weights=clustering_weights,
                                    max_iter=self.max_iter, learning_rate=self.lr,
                                    regularization=1/self.C, adam_beta1=self.adam_beta1,
                                    adam_beta2=self.adam_beta2, eps=self.eps,
                                    tol=self.tol, standardize_descriptive=self.standardize_descriptive,
                                    standardize_clustering=self.standardize_clustering, rng=rng,
                                    objective=self.objective)

        # root_data.d = data.shape[1] + 1 (e.g. Iris has 4 features; root_data.d=5 --> features = [0 1 2 3 4])
        feature_importance = np.zeros(root_data.d-1)
        if num_features == root_data.d-1:
            features = np.arange(root_data.d).astype(np.intp)

        root_data.calc_impurity(self.eps)
        root_node = Node(0)
        
        splitting_queue = [(root_node, root_data, root_data.min_labelled())]
        node_list = [root_node]
        
        odt_info = dict()
        while splitting_queue:
            node, data, num_labelled = splitting_queue.pop()
            
            successful_split = False
            total_impurity = data.total_impurity(clustering_weights)

            if total_impurity > 2 * self.eps and node.depth < self.max_depth and num_labelled >= self.min_examples_to_split:
                    
                # Try to split the node
                if num_features < data.d-1:
                    # data.d-1  ...  number of features
                    # size      ...  max number of features + 1 (for bias)
                    if self.splitting_features is None:
                        features = self.rng.choice(data.d-1, size=num_features+1, replace=False).astype(np.intp)
                    else:
                        temp_features = self.splitting_features.copy()
                        temp_features.append(0)
                        features = np.array(temp_features).astype(np.intp)
                        
                    features[-1] = data.d-1
                    features.sort()                     
                    
                # Gradient-descent-based split learning
                splitter.learn_split(data, features)
                if data.descriptive_data.is_sparse: 
                    split_weights = memview_to_SMatrix(splitter.weights_bias, data.d, features)
                else: 
                    split_weights = memview_to_DMatrix(splitter.weights_bias, data.d, features)
    
                # Only features on which we split, their weights, and a threshold 
                odt_info["features"] = features[:-1]
                odt_info["weights"] = np.asarray(splitter.weights_bias[:-1])
                odt_info["threshold"] = splitter.threshold - splitter.weights_bias[-1]
                
                data_right, data_left = data.split(split_weights, splitter.threshold)
                labelled_left = data_left.min_labelled()
                labelled_right = data_right.min_labelled()
                
                odt_info["n_samples_left"] = labelled_left
                odt_info["n_samples_right"] = labelled_right

                if labelled_left > 0 and labelled_right > 0:
                    data_left.calc_impurity(self.eps)
                    data_right.calc_impurity(self.eps)
                    
                    if relative_impurity(data_left, data, clustering_weights) < self.max_relative_impurity or \
                            relative_impurity(data_right, data, clustering_weights) < self.max_relative_impurity:            
            
                        # We have a useful split!
                        feature_importance[features[:-1]] += splitter.feature_importance

                        if data.descriptive_data.is_sparse: 
                            node.split_weights = memview_to_SMatrix(splitter.weights_bias[:-1], data.d-1, features[:-1])
                        else: 
                            node.split_weights = memview_to_DMatrix(splitter.weights_bias[:-1], data.d-1, features[:-1])
                        
                        node.threshold = splitter.threshold - splitter.weights_bias[-1]
                        
                        left_node = Node(depth=node.depth + 1)
                        node_list.append(left_node)
                        node.left = len(node_list) - 1
                        right_node = Node(depth=node.depth + 1)
                        node_list.append(right_node)
                        node.right = len(node_list) - 1
                        if data.missing_descriptive:
                            feature_means = np.zeros(data.d, dtype=DTYPE)
                            feature_means[features] = splitter.d_means
                            node.feature_means = feature_means
                            
                        successful_split = True

                        splitting_queue.append((right_node, data_right, labelled_right))
                        splitting_queue.append((left_node, data_left, labelled_left))
                        
            if not successful_split:
                
                # Turn the node into a leaf
                if False and self.sparse_target:
                    temp = np.empty(data.t, dtype=DTYPE)
                    data.target_data.column_means(temp)
                    node.prototype = sp.csr_matrix(temp.reshape(1, -1))
                elif data.missing_target:
                    node.prototype = np.empty(data.t, dtype=DTYPE)
                    data.target_data.column_means_nan(node.prototype)
                else:
                    node.prototype = np.empty(data.t, dtype=DTYPE)
                    data.target_data.column_means(node.prototype)


        return np.array(node_list), feature_importance, odt_info

