from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from scipy.special import expit
from copy import deepcopy
import numpy as np
import pandas as pd
import random
import src.odt as odt
from src.utils import check_fit_arguments, leaf_impurity, extract_info, print_split

from src.constants import seed_everything
seed_everything()


class Node:
    """Node class for splitting"""
    
    def __init__(self, features=None, weights=None, threshold: int = None, 
                 value=None, value_sklearn=None, 
                 idxs=None, left=None, right=None,
                 impurity: float=None, impurity_reduction: float=None, 
                 is_root: bool = False, tree_num: int = None):     
        
        # split or linear
        self.is_root = is_root
        self.idxs = idxs
        self.tree_num = tree_num
        # self.node_id = None
        self.impurity = impurity
        self.impurity_reduction = impurity_reduction
        self.value_sklearn = value_sklearn

        # different meanings
        self.value = value  # for split this is mean, for linear this is weight

        # split-specific
        self.features = features
        self.weights = weights
        self.threshold = threshold
        self.left = left
        self.right = right
        self.left_temp = None
        self.right_temp = None
        self.node_id = None
        
        
    def setattrs(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


    def __str__(self):
        if self.is_root:                                    # root
            if isinstance(self.features, int) or isinstance(self.features, np.integer):
                self.features = [self.features]
            out = print_split(self, mode="root")       
            return out
        elif self.left is None and self.right is None:      # leaf
            return f"Val: {self.value[0][0]:0.3f} (leaf)"
        else:                                               # split
            out = print_split(self, mode="split")  
            return out
        
        
    def __repr__(self):
        return self.__str__()

    

class ROFIGS(BaseEstimator):
    def __init__(self,
        beam_size: str = None,                  # max. number of features used per split     
        min_impurity_decrease: float = 0.0,     # node will be split if it induces an impurity decrease that is greater than or equal to this value                             
        max_splits: int = 75,                   # max. total number of splits across all trees            
        max_trees: int = None,                  # max. number of trees
        random_state = None,
        verbose = False,                       
    ):
        
        super().__init__()
        
        self.beam_size = beam_size
        self.min_impurity_decrease = min_impurity_decrease
        self.max_splits = max_splits
        self.max_trees = max_trees
        self.random_state = random_state
        self.verbose = verbose                  # whether we print info during model training

        self.num_repetitions = 5                # 5 tries/repetitions in each iteration
        self.iteration = 1                      # used for changing the subset of features used for splitting 
        self.non_zero_total = 0                 # average number of features per split that are *actually* used for splitting                    
        self.feature_combinations = []          # saving all combinations of features that appear in the model
        
        seed_everything()
        self._init_decision_function()
 

    def _construct_oblique_split(self, X, y, idxs, tree_num, beam_size=2, splitting_features=None): 
        """Create a split by computing a linear combination of splitting_features using an ODT model.
        Args:
            X (_type_): input features
            y (_type_): target
            idxs (_type_): samples in this node
            tree_num (_type_): corresponding tree number
            beam_size (int, optional): number of features considered for splitting; defaults to 2
            splitting_features (_type_, optional): subset of features to split on; defaults to None
        Returns:
            Node: An oblique split node 
        """
        
        SPLIT, LEFT, RIGHT = 0, 1, 2
        
        # create an oblique split 
        stump = odt.Model(max_features=beam_size, splitting_features=splitting_features, random_state=self.random_state)
        y_regr = y.reshape(-1, 1)       
        stump.fit(X[idxs], y_regr[idxs])

        # there's no split on these features at this node -> create a dummy node (i.e., leaf)
        if "error" in stump.odt_info and stump.odt_info["error"]:
            mse_imp = leaf_impurity(X[idxs], y[idxs]) 
            if self.verbose:
                print(f"#samples = {np.sum(idxs)} --> no split was created, return leaf") 
            # no split was created -> we set "features", "weights", and "threshold" to 0
            return Node(features=0, weights=0, threshold=0, value=np.mean(y[idxs]),
                        idxs=idxs, impurity=mse_imp, impurity_reduction=None, tree_num=tree_num)

        # split was created, but all samples belong to one child -> create a dummy node (i.e., leaf)
        if (("n_samples_left" in stump.odt_info) and stump.odt_info["n_samples_left"]==0) or  \
            (("n_samples_right" in stump.odt_info) and stump.odt_info["n_samples_right"]==0):
            mse_imp = leaf_impurity(X[idxs], y[idxs]) 
            if self.verbose:
                print(f"#samples = {np.sum(idxs)} --> no split was created, return leaf") 
            # saving int instead of list with one element if 1-variate split
            feat = stump.odt_info["features"][0] if len(stump.odt_info["features"])==1 else stump.odt_info["features"]   
            return Node(features=feat, weights=stump.odt_info["weights"], threshold=stump.odt_info["threshold"], value=np.mean(y[idxs]),
                        idxs=idxs, impurity=mse_imp, impurity_reduction=None, tree_num=tree_num)
        
        # extract info from the oblique split
        try:
            info_dict = extract_info(stump, X[idxs], y[idxs]) 
        except:
            mse_imp = leaf_impurity(X[idxs], y[idxs]) 
            return Node(features=0, weights=0, threshold=0, value=np.mean(y[idxs]),
                        idxs=idxs, impurity=mse_imp, impurity_reduction=None, tree_num=tree_num)
    
        # if we get to here, the split was successfully created 
        features = info_dict["features"]
        weights = info_dict["weights"]
        threshold = info_dict["threshold"]
        impurity = info_dict["impurity"]
        n_node_samples = info_dict["n_node_samples"]
        value = info_dict["value"]
        
        # samples split
        idxs_split = np.dot(X[:,stump.odt_info["features"]], stump.odt_info["weights"]) <= stump.odt_info["threshold"]
        idxs_left = idxs_split & idxs
        idxs_right = ~idxs_split & idxs
    
        n_node_samples_split = n_node_samples[SPLIT]
        n_node_samples_left = n_node_samples[LEFT]
        n_node_samples_right = n_node_samples[RIGHT]

        # calculate impurity
        impurity_reduction = (impurity[SPLIT] 
                              - impurity[LEFT] * n_node_samples_left / n_node_samples_split
                              - impurity[RIGHT] * n_node_samples_right / n_node_samples_split
                              ) * n_node_samples_split
        if self.verbose:
            print(f"#samples: split = {n_node_samples_split}, left = {n_node_samples_left}, right = {n_node_samples_right}")
            print(f"values: split = {value[0][0][0]:0.3f}, left = {value[1][0][0]:0.3f}, right = {value[2][0][0]:0.3f}")
            print(f"impurity reduction: {impurity_reduction:0.3f}")   

        # define the constructed split and assign children
        node_split = Node(features=features, weights=weights, threshold=threshold, value=value[SPLIT], 
                          idxs=idxs, impurity=impurity[SPLIT], impurity_reduction=impurity_reduction, tree_num=tree_num) 
        node_left = Node(idxs=idxs_left, value=value[LEFT], impurity=impurity[LEFT], tree_num=tree_num)
        node_right = Node(idxs=idxs_right, value=value[RIGHT], impurity=impurity[RIGHT], tree_num=tree_num)
        node_split.setattrs(left_temp=node_left, right_temp=node_right)
        
        return node_split


    def fit(self, X, y=None, feature_names=None, verbose=False):

        X, y, feature_names = check_fit_arguments(self, X, y, feature_names)
        assert self.beam_size <= X.shape[1], "Beam size must be less than or equal to the number of features in the dataset."
        
        self.trees_ = []                # list of the root nodes of added trees
        self.complexity_ = 0            # tracking the number of splits in the model
        y_predictions_per_tree = {}     # predictions for each tree
        y_residuals_per_tree = {}       # residuals

        # set up initial potential_splits
        # everything in potential_splits is either 
                # a root node (flagged with is_root), so that it can be added directly to self.trees_, or
                # a child of a root node that has already been added
        idxs = np.ones(X.shape[0], dtype=bool)
        
        # build the first stump
        if self.verbose:
            print(f"{80*'*'}\n{31*'.'}  Iteration = {self.iteration}  {32*'.'}\n{80*'*'}")
        splitting_features = random.sample([i for i in range(X.shape[1])], self.beam_size) 
        node_init = self._construct_oblique_split(X=X, y=y, idxs=idxs, tree_num=-1, 
                                                  beam_size=self.beam_size, splitting_features=splitting_features)          

        # allow multiple repetitions for the root node   
        num_tries = self.num_repetitions  
        while ((node_init.impurity_reduction is None) and num_tries > 1):
            num_tries -= 1
            if self.verbose:
                print(f">> Number of repetitions left for the root node: {num_tries}")
            splitting_features = random.sample([i for i in range(X.shape[1])], self.beam_size) 
            node_init = self._construct_oblique_split(X=X, y=y, idxs=idxs, tree_num=-1,
                                                      beam_size=self.beam_size, splitting_features=splitting_features)        
        
        # in the first iteration, there's only one potential split - the root of the first tree
        node_init.setattrs(is_root=True)
        potential_splits = [node_init]
        
        # start the greedy fitting algorithm
        finished = False
        while len(potential_splits) > 0 and not finished:
            
            # allow multiple repetitions per iteration  
            num_tries = self.num_repetitions    
            self.iteration += 1
            
            if self.verbose: 
                print(f"\n{80*'*'}\n{31*'.'}  Iteration = {self.iteration}  {32*'.'}\n{80*'*'}")
            
            # get node with max impurity_reduction from the sorted list
            split_node = potential_splits.pop()
                
            # best potential split is not good enough, finish training
            if split_node.impurity_reduction < self.min_impurity_decrease: 
                finished = True
                break
            # this happens if all nodes in the previous iteration were unsuccessfull, 
            # we keep them so that we can try making a split with another subset of features
            elif split_node.impurity_reduction == 0:
                continue
            # don't split on this node if it's a root of a new tree and we have reached self.max_trees
            # however, allow adding other nodes and continue growing the existing trees
            elif split_node.is_root and self.max_trees is not None and len(self.trees_) >= self.max_trees:
                continue

            if self.verbose:
                print(f"Adding a new node with impurity reduction of {split_node.impurity_reduction:0.3f}:\n\t{split_node}")
                
            # add the split to the model
            self.complexity_ += 1
            
            # add the number of features in the root node that have non-zero weights to the total number
            self.non_zero_total += len(np.nonzero(split_node.weights)[0])
            
            # if added a tree root
            if split_node.is_root:
                self.trees_.append(split_node)                                                          # start a new tree
                for node_ in [split_node, split_node.left_temp, split_node.right_temp]:                 # update tree_num
                    if node_ is not None:
                        node_.tree_num = len(self.trees_) - 1   
                node_new_root = Node(is_root=True, idxs=np.ones(X.shape[0], dtype=bool), tree_num=-1)   # add new root potential node        
                potential_splits.append(node_new_root)

            # assign left_temp, right_temp to be proper children and add them to potential_splits
            split_node.setattrs(left=split_node.left_temp, right=split_node.right_temp)
            potential_splits.append(split_node.left)
            potential_splits.append(split_node.right)
        
            # update predictions for altered tree
            for tree_num_ in range(len(self.trees_)):
                y_predictions_per_tree[tree_num_] = self._predict_tree(self.trees_[tree_num_], X)
                
            # dummy 0 predictions for a possible new tree
            y_predictions_per_tree[-1] = np.zeros(X.shape[0])

            # update residuals for each tree; -1 is key for a potential new tree
            for tree_num_ in list(range(len(self.trees_))) + [-1]:
                y_residuals_per_tree[tree_num_] = deepcopy(y)

                # subtract predictions of all other trees; not including -1 because this represent a potential, not yet existing, new tree
                for tree_num_other_ in range(len(self.trees_)):
                    if not tree_num_other_ == tree_num_:
                        y_residuals_per_tree[tree_num_] -= y_predictions_per_tree[tree_num_other_]

            ###
            potential_splits_with_sufficient_impurity_reduction = 0
            
            while num_tries > 0:
                if potential_splits_with_sufficient_impurity_reduction != 0:
                    break     
                
                # recompute all impurities and update potential_split children
                potential_splits_new = []
                
                # choosing the features that are used for splitting in this iteration (and this particular try/repetition)
                # all splits in the same iteration are optimised on the same subset of features
                splitting_features = random.sample([i for i in range(X.shape[1])], self.beam_size) 
                if self.verbose: 
                    print(f"\n>>>>> Splitting on features: {', '.join(map(str, splitting_features))} <<<<<")
                    print(f"\nThere are {len(potential_splits)} potential splits:")

                # re-calculate the best split
                for potential_split in potential_splits:
                
                    if self.verbose: print("\n", 50*"*")
                    
                    # optimise splits
                    y_target = y_residuals_per_tree[potential_split.tree_num]
                    potential_split_updated = self._construct_oblique_split(X=X, y=y_target, idxs=potential_split.idxs, tree_num=potential_split.tree_num, 
                                                                                beam_size=self.beam_size, splitting_features=splitting_features)              
                        
                    # need to preserve certain attributes from before (value at this split + is_root)
                    # value may change because residuals may have changed, but we want it to store the value from before
                    potential_split.setattrs(
                                features=potential_split_updated.features,
                                threshold=potential_split_updated.threshold,
                                weights=potential_split_updated.weights,
                                impurity_reduction=potential_split_updated.impurity_reduction,
                                impurity=potential_split_updated.impurity,
                                left_temp=potential_split_updated.left_temp,
                                right_temp=potential_split_updated.right_temp,
                            )
                        
                    # this is a valid split
                    if potential_split.impurity_reduction is not None:
                        potential_splits_new.append(potential_split)
                    
                    # keep non-valid splits (but set their impurity reduction to 0), so you can split here again with different features
                    else:
                        potential_split.setattrs(impurity_reduction=0)
                        potential_splits_new.append(potential_split)

                # sort potential splits so that the largest impurity reduction comes last
                potential_splits = sorted(potential_splits_new, key=lambda x: x.impurity_reduction)
                
                potential_splits_with_sufficient_impurity_reduction = sum([1 for spl in potential_splits if spl.impurity_reduction > self.min_impurity_decrease])
            
                if self.verbose:
                    print(f"\n{25*'-*'}\n")
                    print(f"Out of {len(potential_splits)} potential split(s), {potential_splits_with_sufficient_impurity_reduction} has/have sufficient impurity reduction (i.e., > {self.min_impurity_decrease}):")
                    for spl in potential_splits:
                        print(f"\t {spl} (imp. red. of {np.round(spl.impurity_reduction,3)})") 
                
                num_tries -= 1
                if self.verbose and potential_splits_with_sufficient_impurity_reduction == 0:
                    print(f"\n{40*'-*'}\n{40*'-*'}")
                    print(f"\n>> Number of repetitions left: {num_tries}")
            ###
            
            # stopping conditions
            if self.max_splits is not None and self.complexity_ >= self.max_splits:
                finished = True
                break        
        
        # annotate final trees with node_id and value_sklearn
        if self.verbose: 
            print("\n\n--Annotation--")
        for tree_ in self.trees_:
            node_counter = iter(range(0, int(1e06)))
            
            def _annotate_node(node: Node, X, y):
                
                if node is None:
                    pass
                elif node.left is None and node.right is None:
                    pass
                else:
                    # saving features with nonzero weights 
                    self.feature_combinations.append(tuple(node.features[node.weights.nonzero()]))
                
                if node is None: # children of leaves
                    return
                value_counts = pd.Series(y).value_counts()
                try:
                    neg_count = value_counts[0.0]
                except KeyError:
                    neg_count = 0
                try:
                    pos_count = value_counts[1.0]
                except KeyError:
                    pos_count = 0
                # how many samples are in classes 0 and 1 (not left and right nodes)
                value_sklearn = np.array([neg_count, pos_count], dtype=float)               
                node.setattrs(node_id=next(node_counter), value_sklearn=value_sklearn)
                    
                # added because for some reason, ODT returned an array with one number instead of just one value
                if not isinstance(node.features, np.ndarray) and isinstance(node.weights, np.ndarray):
                    node.weights = node.weights[0]
                idxs_left = np.dot(X[:,node.features], node.weights) <= node.threshold  
                
                if self.verbose: 
                    # print #samples in the node if leaf, otherwise print #samples in node and both children
                    print_samples = f"\n#samples = {y.shape[0]}" if (node.left is None and node.right is None) else f"\n#samples = {y.shape[0]}; L: {len(X[idxs_left])}, R: {len(X[~idxs_left])}"
                    print(f"\n>>>>> {node} <<<<< {print_samples}")
      
                _annotate_node(node.left, X[idxs_left], y[idxs_left])
                _annotate_node(node.right, X[~idxs_left], y[~idxs_left])
            _annotate_node(tree_, X, y)

        return self
    

    def _tree_to_str(self, root: Node, prefix=""):
        if root is None:
            return ""
        elif root.threshold is None:
            return ""
        pprefix = prefix + "\t"
        return (
            prefix
            + str(root)
            + "\n"
            + self._tree_to_str(root.left, pprefix)
            + self._tree_to_str(root.right, pprefix)
        )

    def __str__(self):
        if not hasattr(self, "trees_"):
            s = self.__class__.__name__
            s += f"(max_splits={repr(self.max_splits)})"
            return s
        else:
            s = "> RO-FIGS:\n"
            s += "\n\t+\n".join([self._tree_to_str(t) for t in self.trees_])
            return s

 
    def get_int_metrics(self):
        num_trees = len(self.trees_)
        num_splits = self.complexity_
        return {"trees": num_trees, 
                "splits": num_splits}


    def count_trees(self):
        return len(self.trees_)
    
    
    def count_splits(self):
        return self.complexity_
    
    
    def get_average_num_feat_per_split(self):
        """Compute the average number of features per split that actually contribute to splitting 
           (i.e., features with non-zero weights)
        """
        num_splits = self.complexity_
        if num_splits > 0:
            return self.non_zero_total / num_splits
        else:
            return 0
    
    
    def predict(self, X):
        X = check_array(X)
        preds = np.zeros(X.shape[0])
        for tree in self.trees_:
            preds += self._predict_tree(tree, X)
        if isinstance(self, RegressorMixin):
            return preds
        elif isinstance(self, ClassifierMixin):
            return (preds > 0.5).astype(int)


    def predict_raw(self, X):
        X = check_array(X)
        preds = np.zeros(X.shape[0])
        for tree in self.trees_:
            preds += self._predict_tree(tree, X)
        return preds

        
    def predict_proba(self, X, use_clipped_prediction=False):
        """Predict probability for classifiers:
        Default behavior is to constrain the outputs to the range of probabilities, i.e. 0 to 1, with a sigmoid function.
        Set use_clipped_prediction=True to use prior behavior of clipping between 0 and 1 instead. Taken from FIGS.
        """
        X = check_array(X)
        if isinstance(self, RegressorMixin):
            return NotImplemented
        preds = np.zeros(X.shape[0])
        for tree in self.trees_:
            preds += self._predict_tree(tree, X)
        if use_clipped_prediction:
            preds = np.clip(preds, a_min=0.0, a_max=1.0)
        else:
            preds = expit(preds)
        return np.vstack((1 - preds, preds)).transpose()


    def _predict_tree(self, root: Node, X):
        """Predict for a single tree"""

        def _predict_tree_single_point(root: Node, x):
            
            # stopping condition
            if root.left is None and root.right is None:
                return root.value[0, 0]

            # unnecessary?; needed when weights and features were stored differently for univariate splits
            if isinstance(root.features, int) or isinstance(root.features, np.integer):
                left = x[root.features] * root.weights <= root.threshold
            else:
                left_temp = (x[root.features[0]] * root.weights[0])
                for feat in range(1, self.beam_size):
                    left_temp += (x[root.features[feat]] * root.weights[feat])
                left = left_temp <= root.threshold
            
            if left:
                if root.left is not None:
                    return _predict_tree_single_point(root.left, x)
            else:
                if root.right is not None:
                    return _predict_tree_single_point(root.right, x)

        preds = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            preds[i] = _predict_tree_single_point(root, X[i])
        return preds

    def _init_decision_function(self):
        """Sets decision function based on _estimator_type"""
        if isinstance(self, ClassifierMixin):
            def decision_function(x):
                return self.predict_proba(x)[:, 1]
        elif isinstance(self, RegressorMixin):
            decision_function = self.predict
            
    @property
    def feature_importances_(self):
        """Gini impurity-based feature importances"""
        check_is_fitted(self)
        avg_feature_importances = np.mean(self.importance_data_, axis=0, dtype=np.float64)
        return avg_feature_importances / np.sum(avg_feature_importances)



class ROFIGSRegressor(ROFIGS, RegressorMixin):
    ...


class ROFIGSClassifier(ROFIGS, ClassifierMixin):
    ...
