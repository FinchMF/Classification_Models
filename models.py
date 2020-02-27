################
# Dependencies #
################

import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
import scipy.stats as stats
import itertools
from sklearn import metrics
import pickle
from time import time
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


#############################
# Plotting Confusion Matrix #
#############################

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion Matrix, without normalization')

    print(cm)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




#######################################
#  Models for Spotify Classifications #
#######################################

# loading and running saved models

def run_model(df, model, classes):
    # set features and target
    X = df.drop(['Genre'], axis = 1)
    y = df['Genre']
    # using standardscaler to normalize
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)
    df = pd.DataFrame(X, columns=X.columns)
    # train test split
    # training/test
    X_train, X_test, y_train, y_test = train_test_split(df, 
                                                        y, 
                                                        test_size=0.2, 
                                                        random_state=42)
    model_score = model.score(X_test, y_test)
    print('Model Accuracy Score:', model_score)
    print('\n')
    # fit model and make predictions
    preds = model.predict(X_test)
    model_confusion_matrix = confusion_matrix(y_test, preds)
    model_classification_report = classification_report(y_test, preds)
    print(model_confusion_matrix)
    print(model_classification_report)
    print('\n')
    print('Classification Results: ')
    print('\n')
    model_results = list(zip(list(preds), list(y_test)))
    print(model_results)
    plot_cm = plot_confusion_matrix(model_confusion_matrix,
                                    classes,
                                    normalize = False,
                                    title = 'Model Parameters: ' + str(model) + '\n' + 'Confusion Matrix',
                                    cmap=plt.cm.Purples)
    print(plot_cm)



# searching and saving optimized model 

def find_model(df, model):
    