import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, accuracy_score
from sklearn.utils import resample

class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def predict(self, x_test):
        if x_test[self.col]<=self.split:
            return self.lchild.predict(x_test)
        else:
            return self.rchild.predict(x_test)
    
    def leaf(self, x_test):
        if x_test[self.col]<=self.split:
            return self.lchild.leaf(x_test)
        else:
            return self.rchild.leaf(x_test)
        

class LeafNode:
    def __init__(self, y, prediction):
        self.n = len(y)
        self.prediction = prediction
        self.class_values = y

    def predict(self, x_test):
        return self.prediction
    
    def leaf(self, x_test):
        return self


def gini(x):
    """
    calculates gini impurity
    """
    tmp, counts_array = np.unique(x, return_counts=True)
    p_i_array = counts_array / len(x)
    return 1 - np.sum( p_i_array**2 )


def find_best_split(X, y, loss, min_samples_leaf, max_features):
    best = (-1, -1, loss(y))
    random_features = np.random.choice(X.shape[1], int(max_features*(X.shape[1])), replace= False)
    for p_col_idx in random_features: #looping over random subset of columns
        X_col = X[:,p_col_idx]
        candidates = X_col[np.random.randint(0,len(X_col), size=11)] # picking 11 random values of the column
        for split in candidates:
            yl = y[X_col<=split]
            yr = y[X_col>split]
            if (len(yl)<min_samples_leaf or len(yr)<min_samples_leaf): continue
            l = ((len(yl)*loss(yl))+(len(yr)*loss(yr)))/len(y)
            if l == 0: return p_col_idx,split
            if l < best[2]: best = (p_col_idx, split, l)
    return best[0], best[1]
    
    
class DecisionTree:
    def __init__(self, min_samples_leaf=1, loss=None, max_features=0.93):
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss # loss function; either np.var for regression or gini for classification
        self.max_features = max_features
        
    def fit(self, X, y):
        self.root = self.fit_(X, y)

    def leaf(self, X_test):
        return [self.root.leaf(i) for i in X_test]

    def fit_(self, X, y):
        if ((X.shape[0]<=self.min_samples_leaf) or (len(np.unique(X))==1)): return self.create_leaf(y)
        col, split = find_best_split(X,y,self.loss,self.min_samples_leaf,self.max_features)
        if col==-1: return self.create_leaf(y)
        lchild = self.fit_( X[X[:,col]<=split], y[X[:,col]<=split])
        rchild = self.fit_( X[X[:,col]>split], y[X[:,col]>split])
        return DecisionNode(col, split, lchild,rchild)

    def predict(self, X_test):
        return [self.root.predict(i) for i in X_test]


class RegressionTree(DecisionTree):
    def __init__(self, min_samples_leaf=1,max_features=0.93):
        super().__init__(min_samples_leaf, loss=np.var, max_features=max_features)

    def score(self, X_test, y_test):
        preds = self.predict(X_test)
        return r2_score(y_test,preds) #Rsquare

    def create_leaf(self, y):
        return LeafNode(y, np.mean(y))


class ClassifierTree(DecisionTree):
    def __init__(self, min_samples_leaf=1,max_features=0.93):
        super().__init__(min_samples_leaf, loss=gini,max_features= max_features)

    def score(self, X_test, y_test):
        preds = self.predict(X_test)
        return accuracy_score(y_test,preds)

    def create_leaf(self, y):
        return LeafNode(y, stats.mode(y)[0][0])


class RandomForest:
    def __init__(self, n_estimators=10, oob_score=False):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan

    def fit(self, X, y):
        rf_trees = {}
        all_data_indices = np.arange(len(X))
        all_oob_indexes = []
        for tree_idx in range(self.n_estimators):            
            sample_indices_for_tree = resample(all_data_indices)
            oob_samples_for_tree = list(set(all_data_indices).difference(set(sample_indices_for_tree)))
            
            dt = self.tree_obj_type(min_samples_leaf=self.min_samples_leaf, max_features=self.max_features)
            dt.fit(X[sample_indices_for_tree],y[sample_indices_for_tree])
            
            # storing the trained tree and oob sample indexes
            rf_trees[tree_idx] = (dt,oob_samples_for_tree)
            all_oob_indexes = all_oob_indexes + oob_samples_for_tree 
        self.trees=rf_trees
        self.nunique=np.max(y)
        self.all_oob_indexes_set = set(all_oob_indexes)
        if self.oob_score:
            self.oob_score_ = self.compute_oob_score( X, y)

class RandomForestRegressor(RandomForest):
    def __init__(self, n_estimators=10, min_samples_leaf=3, 
    max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.trees = np.nan
        self.min_samples_leaf= min_samples_leaf
        self.max_features=max_features
        self.loss=np.var
        self.tree_obj_type = RegressionTree


    def predict(self, X_test) -> np.ndarray:
        all_leaf_methods = [i[0].leaf for i in self.trees.values()]
        preds = []
        for i in X_test:
            sum_i_num,sum_i_denom = 0,0
            for j in all_leaf_methods:
                sum_i_num = sum_i_num + (j([i])[0].n * j([i])[0].prediction)
                sum_i_denom = sum_i_denom+  (j([i])[0].n)
            preds.append(sum_i_num/sum_i_denom)
        return np.array(preds)

        
    def score(self, X_test, y_test) -> float:
        preds = self.predict(X_test)
        return r2_score(y_test,preds)

    def compute_oob_score(self, X_oob, y_oob) -> float:
        all_oob_indexes_list = list(self.all_oob_indexes_set)
        oob_true_preds = y_oob[all_oob_indexes_list]

        oob_preds = []
        for i in all_oob_indexes_list:
            sum_i_num,sum_i_denom = 0,0
            for j in list(self.trees.values()):
                if i in j[1]:
                    oob_leaf = j[0].leaf([X_oob[i]])[0]
                    sum_i_num = sum_i_num+ (oob_leaf.n * oob_leaf.prediction)
                    sum_i_denom = sum_i_denom+  (oob_leaf.n) 
            oob_preds.append(sum_i_num/sum_i_denom)
        return r2_score(oob_true_preds,oob_preds)

class RandomForestClassifier(RandomForest):
    def __init__(self, n_estimators=10, min_samples_leaf=3, 
    max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.trees = np.nan
        self.min_samples_leaf = min_samples_leaf
        self.max_features=max_features
        self.loss=gini
        self.tree_obj_type = ClassifierTree

    def predict(self, X_test) -> np.ndarray:
        all_leaf_methods = [i[0].leaf for i in self.trees.values()]
        preds = []
        for i in X_test:
            class_counts = np.zeros(self.nunique+1)
            for j in all_leaf_methods:
                for c_valuue in j([i])[0].class_values:
                    class_counts[c_valuue]+=1
            preds.append(np.argmax(class_counts))

        return np.array(preds)
        
    def score(self, X_test, y_test) -> float:
        preds = self.predict(X_test)
        return accuracy_score(y_test,preds)
        
    def compute_oob_score(self, X_oob, y_oob) -> float:
        all_oob_indexes_list = list(self.all_oob_indexes_set)
        oob_true_preds = y_oob[all_oob_indexes_list]

        oob_preds = []
        for i in all_oob_indexes_list:
            class_counts = np.zeros(self.nunique+1)
            for j in list(self.trees.values()):
                if i in j[1]:
                    oob_leaf = j[0].leaf([X_oob[i]])[0]
                    for c_valuue in oob_leaf.class_values: class_counts[c_valuue]+=1
            oob_preds.append(np.argmax(class_counts))
            
        return r2_score(oob_true_preds,oob_preds)   