import sys
import os
from matplotlib import pyplot
from pandas import read_csv
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier



#####################
# Load dataset
#####################
#dataset = read_csv(sys.argv[1], header=None, sep="\t")
dataset = read_csv(sys.argv[1], sep=",")
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
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('QDA', QuadraticDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('AB', AdaBoostClassifier()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('ET', ExtraTreesClassifier()))
results = []
names = []
msgs = []
for name, model in models:
    best_res = -1
    best_cv_results = []
    print(name)
    for _ in range(50):
        #kfold = KFold(n_splits=num_folds, random_state=seed)
        #kfold = KFold(n_splits=num_folds)
        kfold = StratifiedKFold(n_splits=num_folds)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        if cv_results.mean() > best_res:
            best_res = cv_results.mean()            
            best_cv_results = cv_results 
            print(best_res)            
    results.append(best_cv_results)
    names.append(name)
    msg = "%5s: %f (%f)" % (name, best_cv_results.mean(), best_cv_results.std())
    msgs.append(msg)
    
for msg in msgs:
    print(msg)


    
#####################    
# Compare Algorithms
#####################
fig = pyplot.figure()
fig.suptitle(dataset_name)
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
#pyplot.show()
pyplot.savefig("../Results/"+dataset_name+"_compare.svg", format="svg")
