import warnings
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree._tree import TREE_LEAF

def learn_local_decision_tree(Z, Yb, weights, class_values, multi_label=False, one_vs_rest=False, cv=5,
                              prune_tree=False):

    dt = DecisionTreeClassifier(
        random_state=42,
        max_depth= 128          # Fixed random seed for reproducibility
      # Consider only a subset of features at each split
    )
    
    if prune_tree:
        param_list = {
            'min_samples_split': [0.01, 0.03, 0.05],
            'min_samples_leaf': [0.01, 0.02, 0.03],
            'max_depth': [16, 32, 48,128],
            'ccp_alpha': [0.0001, 0.001, 0.01] 
        }

        if not multi_label or (multi_label and one_vs_rest):
            if len(class_values) == 2 or (multi_label and one_vs_rest):
                scoring = 'f1'
            else:
                scoring = 'f1_macro'
        else:
            scoring = 'f1_samples'

        try:
            dt_search = GridSearchCV(dt, param_grid=param_list, scoring=scoring, cv=cv, n_jobs=-1)
        except TypeError:
            dt_search = GridSearchCV(dt, param_grid=param_list, scoring=scoring, cv=cv, n_jobs=-1, iid=False)
            
        dt_search.fit(Z, Yb, sample_weight=weights)
        dt = dt_search.best_estimator_
        
        prune_duplicate_leaves(dt)
        
        dt = validate_and_fix_tree(dt)
    else:
        dt.fit(Z, Yb, sample_weight=weights)

    return dt


def is_leaf(inner_tree, index):
    return (inner_tree.children_left[index] == TREE_LEAF and
            inner_tree.children_right[index] == TREE_LEAF)


def prune_index(inner_tree, decisions, index=0):
    if not is_leaf(inner_tree, inner_tree.children_left[index]):
        prune_index(inner_tree, decisions, inner_tree.children_left[index])
    if not is_leaf(inner_tree, inner_tree.children_right[index]):
        prune_index(inner_tree, decisions, inner_tree.children_right[index])

    if (is_leaf(inner_tree, inner_tree.children_left[index]) and
        is_leaf(inner_tree, inner_tree.children_right[index]) and
        (decisions[index] == decisions[inner_tree.children_left[index]]) and
        (decisions[index] == decisions[inner_tree.children_right[index]])):
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF


def prune_duplicate_leaves(dt):
    decisions = dt.tree_.value.argmax(axis=2).flatten().tolist()  # Decision for each node
    prune_index(dt.tree_, decisions)


def validate_and_fix_tree(dt):
    tree_ = dt.tree_
    n_nodes = tree_.node_count
    feature = tree_.feature
    threshold = tree_.threshold
    children_left = tree_.children_left
    children_right = tree_.children_right
    
    # Identify all paths from root to leaf
    def find_paths(node_id, path, paths):
        path = path + [node_id]
        if children_left[node_id] == TREE_LEAF:  # Leaf node
            paths.append(path)
            return
        find_paths(children_left[node_id], path, paths)
        find_paths(children_right[node_id], path, paths)
    
    # Check paths for contradictions
    def has_contradiction(path):
        feature_conditions = {}
        for i in range(len(path) - 1):
            node_id = path[i]
            feat = feature[node_id]
            if feat == -2:
                continue
                
            thr = threshold[node_id]
            if path[i+1] == children_left[node_id]:
                direction = "<="
            else:
                direction = ">"
                
            if feat in feature_conditions:
                existing_conditions = feature_conditions[feat]
                for existing_dir, existing_thr in existing_conditions:
                    # Check for contradiction
                    if (direction == "<=" and existing_dir == ">" and thr >= existing_thr) or \
                       (direction == ">" and existing_dir == "<=" and thr <= existing_thr):
                        return True
                feature_conditions[feat].append((direction, thr))
            else:
                feature_conditions[feat] = [(direction, thr)]
        return False
    
    all_paths = []
    find_paths(0, [], all_paths)
    
    contradiction_found = False
    for path in all_paths:
        if has_contradiction(path):
            contradiction_found = True
            break
    
    if contradiction_found:
        print("Contradictory rules detected! Creating a simpler decision tree...")
        # Try a simpler tree with more regularization
        new_dt = DecisionTreeClassifier(
            max_depth=128,
            min_samples_split=0.05,
            min_samples_leaf=0.025,
            ccp_alpha=0.01,
            random_state=42
        )

        sample_weight = getattr(dt, "_sample_weight", None)
        X = getattr(dt, "_X", None)
        y = getattr(dt, "_y", None)
        
        if X is None or y is None:
            warnings.warn("No access to original training data, returning original tree despite contradictions")
            return dt
            
        new_dt.fit(X, y, sample_weight=sample_weight)
        return new_dt
    
    return dt