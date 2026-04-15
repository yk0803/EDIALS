
# Simplified ILOREM class
import numpy as np
from scipy.spatial.distance import cdist
import torch
from ilore.decision_tree import learn_local_decision_tree
from ilore.explanation import ImageExplanation
from ilore.rule import Condition, Rule, get_counterfactual_rules, get_rule
from utils.pytorch_adversarial import PyTorchImageRandomAdversarialGeneratorLatent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

class ILOREM:
    def __init__(self, bb_predict, class_name, class_values, neigh_type='rnd', ocr=0.1,
                 kernel_width=None, autoencoder=None, use_rgb=True, scale=False, 
                 valid_thr=0.5, filter_crules=True, random_state=0, verbose=False, **kwargs):
        
        self.bb_predict = bb_predict
        self.class_name = class_name
        self.class_values = class_values
        self.neigh_type = neigh_type
        self.filter_crules = bb_predict if filter_crules else None
        self.ocr = ocr
        self.kernel_width = kernel_width
        self.verbose = verbose
        self.random_state = random_state
        self.autoencoder = autoencoder
        self.use_rgb = use_rgb
        self.scale = scale
        self.valid_thr = valid_thr
        
        self.__init_neighbor_fn(kwargs)
    
    def explain_instance(self, img, num_samples=1000, use_weights=True, metric='euclidean'):
        # if self.verbose:
        #     print('Generating neighborhood...')
            
        Z, Yb, class_value = self.neighgen_fn(img, num_samples)
        
        if self.verbose:
            neigh_class, neigh_counts = np.unique(Yb, return_counts=True)
            neigh_class_counts = {self.class_values[k]: v for k, v in zip(neigh_class, neigh_counts)}
            # print(f'Neighborhood class distribution: {neigh_class_counts}')
        
        nbr_features = Z.shape[1]
        self.feature_names = [i for i in range(nbr_features)]
        
        kernel_width = np.sqrt(nbr_features) * 0.75 if self.kernel_width is None else self.kernel_width
        
        weights = None
        if use_weights:
            distances = cdist(Z, Z[0].reshape(1, -1), metric=metric).ravel()
            weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))
        
        # if self.verbose:
        #     print('Learning local decision tree...')
            
        dt = learn_local_decision_tree(Z, Yb, weights, self.class_values, prune_tree=False)
        
        # Make predictions with decision tree
        Yc = dt.predict(Z)
        fidelity = dt.score(Z, Yb, sample_weight=weights)
        
        # if self.verbose:
        #     print(f'Decision tree fidelity: {fidelity:.4f}')
        #     print('Retrieving explanation...')
        
        # Get factual rule
        x = Z[0]
        feature_names = [f"feature_{i}" for i in range(nbr_features)]
        rule = get_rule(x, dt, feature_names, self.class_name, self.class_values, feature_names)
        
        # if self.verbose:
        #     print(f'Factual rule: {rule}')
        
        # Get counterfactual rules
        crules, deltas = get_counterfactual_rules(
            x, Yc[0], dt, Z, Yc, feature_names, self.class_name,
            self.class_values, feature_names, self.autoencoder,
            self.filter_crules
        )
        
        if not crules or len(crules) == 0:
            # if self.verbose:
            #     print('No counterfactual rules found, creating by inverting factual rule...')
                
            target_class = Yc[0]  # Keep the same class
            
            # For each premise in the factual rule, create a counterfactual by inverting it
            artificial_crules = []
            artificial_deltas = []
            
            for premise in rule.premises:
                # Create inverted premise
                inverted_premise = Condition(
                    premise.att,
                    "<=" if premise.op == ">" else ">",
                    premise.thr,
                    premise.is_continuous
                )
                
                # Create rule with the inverted premise targeting the same class
                crule = Rule([inverted_premise], self.class_values[int(target_class)], self.class_name)
                artificial_crules.append(crule)
                
                artificial_deltas.append([inverted_premise])
                
            crules = artificial_crules
            deltas = artificial_deltas
            
            # if self.verbose:
            #     print(f'Created {len(crules)} counterfactual rules by inverting premises')
        
        # Create explanation object
        exp = ImageExplanation(img, self.autoencoder, self.bb_predict, self.neighgen, self.use_rgb)
        exp.bb_pred = Yb[0]
        exp.dt_pred = Yc[0]
        exp.rule = rule
        exp.crules = crules
        exp.deltas = deltas
        exp.dt = dt
        exp.fidelity = fidelity
        exp.Z = Z
        
        return exp
    
    def __init_neighbor_fn(self, kwargs):
        """Initialize the neighborhood generator"""
        if self.neigh_type in ['rnd']:  # random autoencoder
            self.neighgen = PyTorchImageRandomAdversarialGeneratorLatent(
                self.bb_predict, 
                ocr=0.1,
                autoencoder=self.autoencoder,
                min_width=1, 
                min_height=1, 
                scale=self.scale,
                valid_thr=self.valid_thr
            )
        else:
            print('Unknown neighborhood generator')
            raise Exception
        
        self.neighgen_fn = self.neighgen.generate

