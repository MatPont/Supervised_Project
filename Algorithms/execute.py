import sys
import os
from matplotlib import pyplot as plt
from pandas import read_csv
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
from imblearn.under_sampling import NearMiss, NeighbourhoodCleaningRule
from sklearn import preprocessing
from sklearn.metrics import roc_curve, roc_auc_score


under_sampling = RandomUnderSampler()
#under_sampling = ClusterCentroids()
#under_sampling = NearMiss()
#under_sampling = NeighbourhoodCleaningRule()

#####################
# Load dataset
#####################
dataset = read_csv(sys.argv[1], header=None, sep="\t")
#dataset = read_csv(sys.argv[1], sep=",", index_col=0)
print(dataset.shape)
dataset_name = os.path.basename(sys.argv[1])[:-4]

"""quali_names = ["departem", "ptvente", "sitfamil", "csp", "sexer", "codeqlt"]
dataset = dataset.drop(columns=quali_names)"""

#important_features = ["moycred3", "avtscpte", "anciente", "engagemt", "agemvt", "nbcb", "mtfactur", "cartevpr"]
#important_features = ["moycred3", "avtscpte", "anciente", "engagemt", "agemvt", "cartevpr"]

#dataset = dataset[important_features]

array = dataset.values
X = array[:,0:-1].astype(float)
Y = array[:,-1].astype(int)
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
models.append(('SVM', SVC(probability=True)))
models.append(('AB', AdaBoostClassifier()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('ET', ExtraTreesClassifier()))
results = []
names = []
msgs = []
best_prediction = {}
for name, model in models:
    best_res = -1
    best_cv_results = []
    y_probas_pred = []
    print(name)
    for _ in range(50):
        X_train, Y_train = under_sampling.fit_resample(X, Y)
        X_train = preprocessing.scale(X_train)        
        #kfold = KFold(n_splits=num_folds, random_state=seed)
        #kfold = KFold(n_splits=num_folds)
        kfold = StratifiedKFold(n_splits=num_folds)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        if cv_results.mean() > best_res:
            best_res = cv_results.mean()  
            y_probas_pred = cross_val_predict(model,X_train, Y_train, cv=kfold, method="predict_proba")
            best_cv_results = cv_results 
            print(best_res)            
    results.append(best_cv_results)
    names.append(name)
    msg = "%5s: %f (%f)" % (name, best_cv_results.mean(), best_cv_results.std())
    msgs.append(msg)
    best_prediction[name] = y_probas_pred
    
for msg in msgs:
    print(msg)


    
#####################    
# Compare Algorithms
#####################
fig = plt.figure()
fig.suptitle(dataset_name)
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
#plt.show()
plt.savefig("../Results/"+dataset_name+"_compare.svg", format="svg")
plt.close()


#####################    
# Roc Curves
#####################
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], 'k--')

for model_name in best_prediction:
  fpr, tpr, thresholds = roc_curve(Y_train,best_prediction[model_name][:,1])
  plot_roc_curve(fpr,tpr, label=model_name)

plt.legend()
plt.title('ROC')
plt.savefig("roc.svg", format="svg")

for model_name in best_prediction:
    print(model_name, ": ", roc_auc_score(Y_train, best_prediction[model_name][:,1]))
