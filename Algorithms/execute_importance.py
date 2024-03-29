import sys
import os
from matplotlib import pyplot as plt
from pandas import read_csv
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
from imblearn.under_sampling import NearMiss
from sklearn import preprocessing
from sklearn.metrics import roc_curve, roc_auc_score


under_sampling = RandomUnderSampler()
#under_sampling = ClusterCentroids()
under_sampling = NearMiss()

#####################
# Load dataset
#####################
#dataset = read_csv(sys.argv[1], header=None, sep="\t")
dataset = read_csv(sys.argv[1], sep=",", index_col=0)
print(dataset.shape)
dataset_name = os.path.basename(sys.argv[1])[:-4]

quali_names = ["departem", "ptvente", "sitfamil", "csp", "sexer", "codeqlt"]
dataset = dataset.drop(columns=quali_names)

array = dataset.values
X = array[:,0:-1].astype(float)
Y = array[:,-1]
seed = 7
X_train, Y_train = X, Y

print(X_train.shape)
print(Y_train.shape)



#####################
# Evaluate Algorithms
#####################

# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []

models.append(('CART', DecisionTreeClassifier()))
models.append(('AB', AdaBoostClassifier()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('ET', ExtraTreesClassifier()))

num_it = 50

all_importance = np.zeros(len(dataset.columns)-1)

def create_plot(overall_importance, name):
    importance_order = np.flip(np.argsort(overall_importance))
    ordered_importance = overall_importance[importance_order]
    ordered_features = dataset.columns[importance_order]       
    
    print(ordered_features)
    print(ordered_importance)
    
    plt.bar(ordered_features[:7], ordered_importance[:7])
    plt.xticks(rotation=17, fontsize=14)
    plt.savefig("../Results/"+dataset_name+"_"+name+"_features.svg", format="svg")
    plt.close()

for name, model in models:
    print("=====================")
    print(name)
    print("=====================")    
    
    overall_importance = np.zeros(len(dataset.columns)-1)
    
    for _ in range(num_it):
        X_train, Y_train = under_sampling.fit_resample(X, Y)
        X_train = preprocessing.scale(X_train)
        
        model.fit(X_train, Y_train)
       
        overall_importance += model.feature_importances_
    
    overall_importance /= num_it
    all_importance += overall_importance           
    
    create_plot(overall_importance, name)
    
print("=====================")
all_importance /= len(models)
create_plot(all_importance, "average")
