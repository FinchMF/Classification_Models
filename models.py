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
        pass

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

def run_model(df, set_, model, classes):
    # all features
    if set_ == 0:
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
        plot_confusion_matrix(model_confusion_matrix,
                              classes,
                              normalize = False,
                              title = 'Model Parameters: ' + str(model) + '\n' + 'Confusion Matrix',
                              cmap=plt.cm.Purples)

    # only spotify's features
    if set_ == 1:
        # set features and target
        X = df[['acousticness', 
                'danceability', 
                'energy', 
                'instrumentalness', 
                'liveness',
                'loudness',
                'speechiness',
                'valence',
                'Minor', 
                'Major']]
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
        plot_confusion_matrix(model_confusion_matrix,
                              classes,
                              normalize = False,
                              title = 'Model Parameters: ' + str(model) + '\n' + 'Confusion Matrix',
                              cmap=plt.cm.Purples)
        
    # only engineered features
    if set_ == 2:
        # set features and target
        X = df.drop(['acousticness', 
                     'danceability', 
                     'energy', 
                     'instrumentalness', 
                     'liveness',
                     'loudness',
                     'speechiness',
                     'valence',
                     'Minor', 
                     'Major',
                     'Genre'], axis = 1)
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
        plot_confusion_matrix(model_confusion_matrix,
                              classes,
                              normalize = False,
                              title = 'Model Parameters: ' + str(model) + '\n' + 'Confusion Matrix',
                              cmap=plt.cm.Purples)
        




###########################################
# Functions needed for model optimization #
###########################################


# retreiving metrics
def print_metrics(labels, preds):
    print("Precision Score: {}".format(precision_score(labels, preds, average='weighted')))
    print("Recall Score: {}".format(recall_score(labels, preds, average='weighted')))
    print("Accuracy Score: {}".format(accuracy_score(labels, preds)))
    print("F1 Score: {}".format(f1_score(labels, preds, average='weighted')))


#######
# KNN #
#######

# retreiving best K value
def find_best_k(X_train, y_train, X_test, y_test, min_k=1, max_k=25):
    best_k = 0
    best_score = 0.0
    for k in range(min_k, max_k+1, 2):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        f1 = f1_score(y_test, preds, average='micro')
        if f1 > best_score:
            best_k = k
            best_score = f1
    
    print("Best Value for k: {}".format(best_k))
    print("F1-Score: {}".format(best_score))
    return best_k    


#################
# Random Forest #
#################

def plot_feature_importances(model):
    n_features = X_train.shape[1]
    plt.figure(figsize=(8,8))
    plt.barh(range(n_features), model.feature_importances_, align='center', color='purple') 
    plt.yticks(np.arange(n_features), X_train.columns.values) 
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")




# searching and saving optimized model 

def find_model(df, set_, model, classes):
    # all features 
    if set_ == 0:
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
        if model == 'KNN':
            # retriving best K value
            k = find_best_k(X_train, y_train, 
                            X_test, y_test, 
                            min_k=1, max_k=25)
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            preds = knn.predict(X_test)
            KNN_confusion_matrix = confusion_matrix(y_test, preds)
            KNN_classification_report = classification_report(y_test, preds)
            print(KNN_confusion_matrix)
            print(KNN_classification_report)
            print_metrics(y_test, preds)
            print('\n')
            KNN_results = list(zip(list(preds), list(y_test)))
            print('Classification Results:')
            print(KNN_results)
            plot_confusion_matrix(KNN_confusion_matrix, classes,
                                  normalize=False,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Blues)   
                    
            filename = './models/KNN_optimized_model.sav'
            pickle.dump(knn, open(filename, 'wb'))
            print('Optimized Model Saved')
        
        if model == 'Random_Forest':
            print('Random Forst without GridSearch')
            forest = RandomForestClassifier(n_estimators=300, max_depth= 60)
            forest.fit(X_train, y_train)
            preds = forest.predict(X_test)
            random_forest_confusion_matrix = confusion_matrix(y_test, preds)
            random_forest_classification_report = classification_report(y_test, preds)
            print(random_forest_confusion_matrix)
            print(random_forest_classification_report)
            print("The accuracy score is" + " "+ str(accuracy_score(y_test, preds)))
            print_metrics(y_test, preds)
            random_forest_results = list(zip(list(preds), list(y_test)))
            print('\n')
            print('Classification Results:')
            print(random_forest_results)
            plot_confusion_matrix(random_forest_confusion_matrix, classes,
                                  normalize=False,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Blues)
            filename = './models/Random_Forest_non_optimized_model.sav'
            pickle.dump(forest, open(filename, 'wb'))
            print('Non Optimized Model Saved.....Now performing GridSearch for Optimization')
            rf_clf = RandomForestClassifier()
            mean_rf_cv_score = np.mean(cross_val_score(rf_clf, X_train, y_train, cv=10))
            print(f"Mean Cross Validation Score for Random Forest Classifier: {mean_rf_cv_score :.2%}")
            rf_param_grid = {
                            'n_estimators': [10, 30, 100],
                            'criterion': ['gini', 'entropy'],
                            'max_depth': [None, 2, 6, 10],
                            'min_samples_split': [5, 10],
                            'min_samples_leaf': [3, 6]}
            rf_grid_search = GridSearchCV(rf_clf, 
                                        rf_param_grid, 
                                        cv=10)
            rf_grid_search.fit(X_train, y_train)
            print('Optimization Completed')
            print(f"Training Accuracy: {rf_grid_search.best_score_ :.2%}")
            print("")
            print(f"Optimal Parameters: {rf_grid_search.best_params_}")

            params = rf_grid_search.best_params_
            forest = RandomForestClassifier(criterion= params.get('criterion'), 
                                            n_estimators= params.get('n_estimators'), 
                                            max_depth= params.get('max_depth'),
                                            min_samples_leaf= params.get('min_samples_leaf')
                                            )
            forest.fit(X_train, y_train)
            preds = forest.predict(X_test)
            random_forest_confusion_matrix = confusion_matrix(y_test, preds)
            random_forest_classification_report = classification_report(y_test, preds)
            print(random_forest_confusion_matrix)
            print(random_forest_classification_report)
            print("The accuracy score is" + " "+ str(accuracy_score(y_test, preds)))
            print_metrics(y_test, preds)
            random_forest_results = list(zip(list(preds), list(y_test)))
            print(random_forest_results)
            plot_confusion_matrix(random_forest_confusion_matrix, classes,
                                  normalize=False,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Purples)
            filename = './models/Random_Forest_Optimized_model_.sav'
            pickle.dump(forest, open(filename, 'wb'))
            print('Optimized model saved')

        if model == 'SVC':
            tic = time()
            svclassifier = SVC(kernel='rbf', C=1.906667, degree = 6)  
            svclassifier.fit(X_train, y_train) 
            preds = svclassifier.predict(X_test)
            toc = time()
            print("run time is {} seconds".format(toc-tic))
            SVC_confusion_matrix = confusion_matrix(y_test,preds)
            SVC_classification_report = classification_report(y_test,preds)
            print(SVC_confusion_matrix)
            print(SVC_classification_report)
            print("The accuracy score is" + " "+ str(accuracy_score(y_test, preds)))
            print_metrics(y_test, preds)
            print('\n')
            print('Classification Results:')
            svc_results = list(zip(list(preds), list(y_test)))
            print(svc_results)
            plot_cm = plot_confusion_matrix(SVC_confusion_matrix, classes,
                                            normalize=False,
                                            title='Confusion matrix',
                                            cmap=plt.cm.Blues)
            print(plot_cm)
            filename = './models/SVC_Non_Optimized_model.sav'
            pickle.dump(svclassifier, open(filename, 'wb'))
            print('Non Optimized Model Saved.....Now performing GridSearch for Optimization')
            # using gridsearch to find optimal parameters
            tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                                'C': [1, 10, 100, 1000]},
                                {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

            scores = ['precision', 'recall']

            for score in scores:
                print("# Tuning hyper-parameters for %s" % score)
                print()

                clf = GridSearchCV(
                    SVC(), tuned_parameters, scoring='%s_macro' % score
                )
                clf.fit(X_train, y_train)

                print("Best parameters set found on development set:")
                print()
                print(clf.best_params_)
                print()
                print("Grid scores on development set:")
                print()
                means = clf.cv_results_['mean_test_score']
                stds = clf.cv_results_['std_test_score']
                for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                    print("%0.3f (+/-%0.03f) for %r"
                        % (mean, std * 2, params))
                print()

                print("Detailed classification report:")
                print()
                print("The model is trained on the full development set.")
                print("The scores are computed on the full evaluation set.")
                print()
                y_true, y_pred = y_test, clf.predict(X_test)
                print(classification_report(y_true, y_pred))
                print(confusion_matrix(y_true, y_pred))
                print()
            params = clf.best_params_
            tic = time()
            svclassifier = SVC(kernel=params.get('kernel'), 
                                C=params.get('C'),
                                gamma=params.get('gamma'))  
            svclassifier.fit(X_train, y_train) 
            preds = svclassifier.predict(X_test)
            toc = time()
            print("run time is {} seconds".format(toc-tic)) 
            SVC_confusion_matrix = confusion_matrix(y_test,preds)
            SVC_classification_report = classification_report(y_test,preds)
            print(SVC_confusion_matrix)
            print(SVC_classification_report)
            print("The accuracy score is" + " "+ str(accuracy_score(y_test, preds)))
            print('\n')
            print('Classification Results:')
            svc_results = list(zip(list(preds), list(y_test)))
            print(svc_results)
            plot_confusion_matrix(SVC_confusion_matrix, classes,
                                  normalize=False,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Purples)
            filename = './models/SVC_Optimized_model.sav'
            pickle.dump(svclassifier, open(filename, 'wb'))
            print('Optimized Model Saved')
        
        if model == 'Logistical Regression':
            logreg = LogisticRegression().fit(X_train,y_train)
            preds = logreg.predict(X_test)
            logreg_confusion_matrix = confusion_matrix(y_test,preds)
            logreg_classification_report = classification_report(y_test,preds)
            print(logreg_confusion_matrix)
            print(logreg_classification_report)
            print("The accuracy score is" + " "+ str(accuracy_score(y_test, preds)))
            print_metrics(y_test, preds)
            print('\n')
            print('Classification Results:')
            logreg_results = list(zip(list(preds), list(y_test)))
            print(logreg_results)
            plot_confusion_matrix(logreg_confusion_matrix, classes,
                                  normalize=False,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Purples)
            filename = './models/logreg_non_opitmized.sav'
            pickle.dump(logreg, open(filename, 'wb'))
            print('Non Optimized Model Saved...Now performing GridSearch for Optimization')

            penalty = ['l1', 'l2']
            C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
            solver = ['liblinear', 'saga']

            param_grid = dict(penalty=penalty,
                            C=C,
                            solver=solver)

            grid = GridSearchCV(estimator=logreg,
                                param_grid=param_grid,
                                verbose=1,
                                n_jobs=-1)
            grid_result = grid.fit(X_train, y_train)

            print('Best Score: ', grid_result.best_score_)
            print('Best Params: ', grid_result.best_params_)
            params = grid_result.best_params_
            logreg_1 = LogisticRegression(C = params.get('C'), 
                                        penalty = params.get('penalty'), 
                                        solver= params.get('solver'))
            logreg_1.fit(X_train, y_train)
            pred = logreg_1.predict(X_test)
            logreg_1_confusion_matrix = confusion_matrix(y_test,preds)
            logreg_1_classification_report = classification_report(y_test,preds)
            print(logreg_confusion_matrix)
            print(logreg_classification_report)
            print("The accuracy score is" + " "+ str(accuracy_score(y_test, preds)))
            logreg_1_results = list(zip(list(preds), list(y_test)))
            print('\n')
            print('Classification Results:')
            print(logreg_1_results)
            plot_confusion_matrix(logreg_1_confusion_matrix, classes,
                                  normalize=False,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Purples)
            filename = './models/logreg_optimized.sav'
            pickle.dump(logreg_1, open(filename, 'wb'))
            print('Optimzed Model Saved')
    
    # only spotify's features
    if set_ == 1:
        # set features and target
        X = df[['acousticness', 
                'danceability', 
                'energy', 
                'instrumentalness', 
                'liveness',
                'loudness',
                'speechiness',
                'valence',
                'Minor', 
                'Major']]
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
        if model == 'KNN':
            # retriving best K value
            k = find_best_k(X_train, y_train, 
                            X_test, y_test, 
                            min_k=1, max_k=25)
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            preds = knn.predict(X_test)
            KNN_confusion_matrix = confusion_matrix(y_test, preds)
            KNN_classification_report = classification_report(y_test, preds)
            print(KNN_confusion_matrix)
            print(KNN_classification_report)
            print_metrics(y_test, preds)
            print('\n')
            KNN_results = list(zip(list(preds), list(y_test)))
            print('Classification Results:')
            print(KNN_results)
            plot_confusion_matrix(KNN_confusion_matrix, classes,
                                  normalize=False,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Blues)           
            filename = './models/KNN_optimized_model.sav'
            pickle.dump(knn, open(filename, 'wb'))
            print('Optimized Model Saved')
        
        if model == 'Random_Forest':
            print('Random Forst without GridSearch')
            forest = RandomForestClassifier(n_estimators=300, max_depth= 60)
            forest.fit(X_train, y_train)
            preds = forest.predict(X_test)
            random_forest_confusion_matrix = confusion_matrix(y_test, preds)
            random_forest_classification_report = classification_report(y_test, preds)
            print(random_forest_confusion_matrix)
            print(random_forest_classification_report)
            print("The accuracy score is" + " "+ str(accuracy_score(y_test, preds)))
            print_metrics(y_test, preds)
            random_forest_results = list(zip(list(preds), list(y_test)))
            print('\n')
            print('Classification Results:')
            print(random_forest_results)
            plot_confusion_matrix(random_forest_confusion_matrix, classes,
                                  normalize=False,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Blues)
            filename = './models/Random_Forest_non_optimized_model.sav'
            pickle.dump(forest, open(filename, 'wb'))
            print('Non Optimized Model Saved.....Now performing GridSearch for Optimization')
            rf_clf = RandomForestClassifier()
            mean_rf_cv_score = np.mean(cross_val_score(rf_clf, X_train, y_train, cv=10))
            print(f"Mean Cross Validation Score for Random Forest Classifier: {mean_rf_cv_score :.2%}")
            rf_param_grid = {
                            'n_estimators': [10, 30, 100],
                            'criterion': ['gini', 'entropy'],
                            'max_depth': [None, 2, 6, 10],
                            'min_samples_split': [5, 10],
                            'min_samples_leaf': [3, 6]}
            rf_grid_search = GridSearchCV(rf_clf, 
                                        rf_param_grid, 
                                        cv=10)
            rf_grid_search.fit(X_train, y_train)
            print('Optimization Completed')
            print(f"Training Accuracy: {rf_grid_search.best_score_ :.2%}")
            print("")
            print(f"Optimal Parameters: {rf_grid_search.best_params_}")

            params = rf_grid_search.best_params_
            forest = RandomForestClassifier(criterion= params.get('criterion'), 
                                            n_estimators= params.get('n_estimators'), 
                                            max_depth= params.get('max_depth'),
                                            min_samples_leaf= params.get('min_samples_leaf')
                                            )
            forest.fit(X_train, y_train)
            preds = forest.predict(X_test)
            random_forest_confusion_matrix = confusion_matrix(y_test, preds)
            random_forest_classification_report = classification_report(y_test, preds)
            print(random_forest_confusion_matrix)
            print(random_forest_classification_report)
            print("The accuracy score is" + " "+ str(accuracy_score(y_test, preds)))
            print_metrics(y_test, preds)
            random_forest_results = list(zip(list(preds), list(y_test)))
            print(random_forest_results)
            plot_confusion_matrix(random_forest_confusion_matrix, classes,
                                  normalize=False,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Purples)
            filename = './models/Random_Forest_Optimized_model_.sav'
            pickle.dump(forest, open(filename, 'wb'))
            print('Optimized model saved')

        if model == 'SVC':
            tic = time()
            svclassifier = SVC(kernel='rbf', C=1.906667, degree = 6)  
            svclassifier.fit(X_train, y_train) 
            preds = svclassifier.predict(X_test)
            toc = time()
            print("run time is {} seconds".format(toc-tic))
            SVC_confusion_matrix = confusion_matrix(y_test,preds)
            SVC_classification_report = classification_report(y_test,preds)
            print(SVC_confusion_matrix)
            print(SVC_classification_report)
            print("The accuracy score is" + " "+ str(accuracy_score(y_test, preds)))
            print_metrics(y_test, preds)
            print('\n')
            print('Classification Results:')
            svc_results = list(zip(list(preds), list(y_test)))
            print(svc_results)
            plot_confusion_matrix(SVC_confusion_matrix, classes,
                                  normalize=False,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Blues)
            filename = './models/SVC_Non_Optimized_model.sav'
            pickle.dump(svclassifier, open(filename, 'wb'))
            print('Non Optimized Model Saved.....Now performing GridSearch for Optimization')
            # using gridsearch to find optimal parameters
            tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                                'C': [1, 10, 100, 1000]},
                                {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

            scores = ['precision', 'recall']

            for score in scores:
                print("# Tuning hyper-parameters for %s" % score)
                print()

                clf = GridSearchCV(
                    SVC(), tuned_parameters, scoring='%s_macro' % score
                )
                clf.fit(X_train, y_train)

                print("Best parameters set found on development set:")
                print()
                print(clf.best_params_)
                print()
                print("Grid scores on development set:")
                print()
                means = clf.cv_results_['mean_test_score']
                stds = clf.cv_results_['std_test_score']
                for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                    print("%0.3f (+/-%0.03f) for %r"
                        % (mean, std * 2, params))
                print()

                print("Detailed classification report:")
                print()
                print("The model is trained on the full development set.")
                print("The scores are computed on the full evaluation set.")
                print()
                y_true, y_pred = y_test, clf.predict(X_test)
                print(classification_report(y_true, y_pred))
                print(confusion_matrix(y_true, y_pred))
                print()
            params = clf.best_params_
            tic = time()
            svclassifier = SVC(kernel=params.get('kernel'), 
                               C=params.get('C'),
                               gamma=params.get('gamma') )  
            svclassifier.fit(X_train, y_train) 
            preds = svclassifier.predict(X_test)
            toc = time()
            print("run time is {} seconds".format(toc-tic)) 
            SVC_confusion_matrix = confusion_matrix(y_test,preds)
            SVC_classification_report = classification_report(y_test,preds)
            print(SVC_confusion_matrix)
            print(SVC_classification_report)
            print("The accuracy score is" + " "+ str(accuracy_score(y_test, preds)))
            print('\n')
            print('Classification Results:')
            svc_results = list(zip(list(preds), list(y_test)))
            print(svc_results)
            plot_confusion_matrix(SVC_confusion_matrix, classes,
                                  normalize=False,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Purples)
            filename = './models/SVC_Optimized_model.sav'
            pickle.dump(svclassifier, open(filename, 'wb'))
            print('Optimized Model Saved')
        
        if model == 'Logistical Regression':
            logreg = LogisticRegression().fit(X_train,y_train)
            preds = logreg.predict(X_test)
            logreg_confusion_matrix = confusion_matrix(y_test,preds)
            logreg_classification_report = classification_report(y_test,preds)
            print(logreg_confusion_matrix)
            print(logreg_classification_report)
            print("The accuracy score is" + " "+ str(accuracy_score(y_test, preds)))
            print_metrics(y_test, preds)
            print('\n')
            print('Classification Results:')
            logreg_results = list(zip(list(preds), list(y_test)))
            print(logreg_results)
            plot_confusion_matrix(logreg_confusion_matrix, classes,
                                  normalize=False,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Purples)
            filename = './models/logreg_non_opitmized.sav'
            pickle.dump(logreg, open(filename, 'wb'))
            print('Non Optimized Model Saved...Now performing GridSearch for Optimization')

            penalty = ['l1', 'l2']
            C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
            solver = ['liblinear', 'saga']

            param_grid = dict(penalty=penalty,
                              C=C,
                              solver=solver)

            grid = GridSearchCV(estimator=logreg,
                                param_grid=param_grid,
                                verbose=1,
                                n_jobs=-1)
            grid_result = grid.fit(X_train, y_train)

            print('Best Score: ', grid_result.best_score_)
            print('Best Params: ', grid_result.best_params_)
            params = grid_result.best_params_
            logreg_1 = LogisticRegression(C = params.get('C'), 
                                          penalty = params.get('penalty'), 
                                          solver= params.get('solver'))
            logreg_1.fit(X_train, y_train)
            pred = logreg_1.predict(X_test)
            logreg_1_confusion_matrix = confusion_matrix(y_test,preds)
            logreg_1_classification_report = classification_report(y_test,preds)
            print(logreg_confusion_matrix)
            print(logreg_classification_report)
            print("The accuracy score is" + " "+ str(accuracy_score(y_test, preds)))
            logreg_1_results = list(zip(list(preds), list(y_test)))
            print('\n')
            print('Classification Results:')
            print(logreg_1_results)
            plot_confusion_matrix(logreg_1_confusion_matrix, classes,
                                  normalize=False,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Purples)
            filename = './models/logreg_optimized.sav'
            pickle.dump(logreg_1, open(filename, 'wb'))
            print('Optimzed Model Saved')        

    # only engineered features
    if set_ == 2:
        # set features and target
        X = df.drop(['acousticness', 
                     'danceability', 
                     'energy', 
                     'instrumentalness', 
                     'liveness',
                     'loudness',
                     'speechiness',
                     'valence',
                     'Minor', 
                     'Major',
                     'Genre'], axis = 1)
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
        if model == 'KNN':
            # retriving best K value
            k = find_best_k(X_train, y_train, 
                            X_test, y_test, 
                            min_k=1, max_k=25)
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            preds = knn.predict(X_test)
            KNN_confusion_matrix = confusion_matrix(y_test, preds)
            KNN_classification_report = classification_report(y_test, preds)
            print(KNN_confusion_matrix)
            print(KNN_classification_report)
            print_metrics(y_test, preds)
            print('\n')
            KNN_results = list(zip(list(preds), list(y_test)))
            print('Classification Results:')
            print(KNN_results)
            plot_confusion_matrix(KNN_confusion_matrix, classes,
                                  normalize=False,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Blues)          
            filename = './models/KNN_optimized_model.sav'
            pickle.dump(knn, open(filename, 'wb'))
            print('Optimized Model Saved')
        
        if model == 'Random_Forest':
            print('Random Forst without GridSearch')
            forest = RandomForestClassifier(n_estimators=300, max_depth= 60)
            forest.fit(X_train, y_train)
            preds = forest.predict(X_test)
            random_forest_confusion_matrix = confusion_matrix(y_test, preds)
            random_forest_classification_report = classification_report(y_test, preds)
            print(random_forest_confusion_matrix)
            print(random_forest_classification_report)
            print("The accuracy score is" + " "+ str(accuracy_score(y_test, preds)))
            print_metrics(y_test, preds)
            random_forest_results = list(zip(list(preds), list(y_test)))
            print('\n')
            print('Classification Results:')
            print(random_forest_results)
            plot_confusion_matrix(random_forest_confusion_matrix, classes,
                                  normalize=False,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Blues)
            filename = './models/Random_Forest_non_optimized_model.sav'
            pickle.dump(forest, open(filename, 'wb'))
            print('Non Optimized Model Saved.....Now performing GridSearch for Optimization')
            rf_clf = RandomForestClassifier()
            mean_rf_cv_score = np.mean(cross_val_score(rf_clf, X_train, y_train, cv=10))
            print(f"Mean Cross Validation Score for Random Forest Classifier: {mean_rf_cv_score :.2%}")
            rf_param_grid = {
                            'n_estimators': [10, 30, 100],
                            'criterion': ['gini', 'entropy'],
                            'max_depth': [None, 2, 6, 10],
                            'min_samples_split': [5, 10],
                            'min_samples_leaf': [3, 6]
                            }
            rf_grid_search = GridSearchCV(rf_clf, 
                                          rf_param_grid, 
                                          cv=10)
            rf_grid_search.fit(X_train, y_train)
            print('Optimization Completed')
            print(f"Training Accuracy: {rf_grid_search.best_score_ :.2%}")
            print("")
            print(f"Optimal Parameters: {rf_grid_search.best_params_}")

            params = rf_grid_search.best_params_
            forest = RandomForestClassifier(criterion= params.get('criterion'), 
                                            n_estimators= params.get('n_estimators'), 
                                            max_depth= params.get('max_depth'),
                                            min_samples_leaf= params.get('min_samples_leaf')
                                            )
            forest.fit(X_train, y_train)
            preds = forest.predict(X_test)
            random_forest_confusion_matrix = confusion_matrix(y_test, preds)
            random_forest_classification_report = classification_report(y_test, preds)
            print(random_forest_confusion_matrix)
            print(random_forest_classification_report)
            print("The accuracy score is" + " "+ str(accuracy_score(y_test, preds)))
            print_metrics(y_test, preds)
            random_forest_results = list(zip(list(preds), list(y_test)))
            print(random_forest_results)
            plot_confusion_matrix(random_forest_confusion_matrix, classes,
                                  normalize=False,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Purples)
            filename = './models/Random_Forest_Optimized_model_.sav'
            pickle.dump(forest, open(filename, 'wb'))
            print('Optimized model saved')

        if model == 'SVC':
            tic = time()
            svclassifier = SVC(kernel='rbf', C=1.906667, degree = 6)  
            svclassifier.fit(X_train, y_train) 
            preds = svclassifier.predict(X_test)
            toc = time()
            print("run time is {} seconds".format(toc-tic))
            SVC_confusion_matrix = confusion_matrix(y_test,preds)
            SVC_classification_report = classification_report(y_test,preds)
            print(SVC_confusion_matrix)
            print(SVC_classification_report)
            print("The accuracy score is" + " "+ str(accuracy_score(y_test, preds)))
            print_metrics(y_test, preds)
            print('\n')
            print('Classification Results:')
            svc_results = list(zip(list(preds), list(y_test)))
            print(svc_results)
            plot_confusion_matrix(SVC_confusion_matrix, classes,
                                  normalize=False,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Blues)
            filename = './models/SVC_Non_Optimized_model.sav'
            pickle.dump(svclassifier, open(filename, 'wb'))
            print('Non Optimized Model Saved.....Now performing GridSearch for Optimization')
            # using gridsearch to find optimal parameters
            tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                                'C': [1, 10, 100, 1000]},
                                {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

            scores = ['precision', 'recall']

            for score in scores:
                print("# Tuning hyper-parameters for %s" % score)
                print()

                clf = GridSearchCV(
                    SVC(), tuned_parameters, scoring='%s_macro' % score
                )
                clf.fit(X_train, y_train)

                print("Best parameters set found on development set:")
                print()
                print(clf.best_params_)
                print()
                print("Grid scores on development set:")
                print()
                means = clf.cv_results_['mean_test_score']
                stds = clf.cv_results_['std_test_score']
                for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                    print("%0.3f (+/-%0.03f) for %r"
                        % (mean, std * 2, params))
                print()

                print("Detailed classification report:")
                print()
                print("The model is trained on the full development set.")
                print("The scores are computed on the full evaluation set.")
                print()
                y_true, y_pred = y_test, clf.predict(X_test)
                print(classification_report(y_true, y_pred))
                print(confusion_matrix(y_true, y_pred))
                print()
            params = clf.best_params_
            tic = time()
            svclassifier = SVC(kernel=params.get('kernel'), 
                               C=params.get('C'),
                               gamma=params.get('gamma'))  
            svclassifier.fit(X_train, y_train) 
            preds = svclassifier.predict(X_test)
            toc = time()
            print("run time is {} seconds".format(toc-tic)) 
            SVC_confusion_matrix = confusion_matrix(y_test,preds)
            SVC_classification_report = classification_report(y_test,preds)
            print(SVC_confusion_matrix)
            print(SVC_classification_report)
            print("The accuracy score is" + " "+ str(accuracy_score(y_test, preds)))
            print('\n')
            print('Classification Results:')
            svc_results = list(zip(list(preds), list(y_test)))
            print(svc_results)
            plot_confusion_matrix(SVC_confusion_matrix, classes,
                                  normalize=False,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Purples)
            filename = './models/SVC_Optimized_model.sav'
            pickle.dump(svclassifier, open(filename, 'wb'))
            print('Optimized Model Saved')
        
        if model == 'Logistical Regression':
            logreg = LogisticRegression().fit(X_train,y_train)
            preds = logreg.predict(X_test)
            logreg_confusion_matrix = confusion_matrix(y_test,preds)
            logreg_classification_report = classification_report(y_test,preds)
            print(logreg_confusion_matrix)
            print(logreg_classification_report)
            print("The accuracy score is" + " "+ str(accuracy_score(y_test, preds)))
            print_metrics(y_test, preds)
            print('\n')
            print('Classification Results:')
            logreg_results = list(zip(list(preds), list(y_test)))
            print(logreg_results)
            plot_confusion_matrix(logreg_confusion_matrix, classes,
                                  normalize=False,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Purples)
            filename = './models/logreg_non_opitmized.sav'
            pickle.dump(logreg, open(filename, 'wb'))
            print('Non Optimized Model Saved...Now performing GridSearch for Optimization')

            penalty = ['l1', 'l2']
            C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
            solver = ['liblinear', 'saga']

            param_grid = dict(penalty=penalty,
                              C=C,
                              solver=solver)

            grid = GridSearchCV(estimator=logreg,
                                param_grid=param_grid,
                                verbose=1,
                                n_jobs=-1)
            grid_result = grid.fit(X_train, y_train)

            print('Best Score: ', grid_result.best_score_)
            print('Best Params: ', grid_result.best_params_)
            params = grid_result.best_params_
            logreg_1 = LogisticRegression(C = params.get('C'), 
                                        penalty = params.get('penalty'), 
                                        solver= params.get('solver'))
            logreg_1.fit(X_train, y_train)
            pred = logreg_1.predict(X_test)
            logreg_1_confusion_matrix = confusion_matrix(y_test,preds)
            logreg_1_classification_report = classification_report(y_test,preds)
            print(logreg_confusion_matrix)
            print(logreg_classification_report)
            print("The accuracy score is" + " "+ str(accuracy_score(y_test, preds)))
            logreg_1_results = list(zip(list(preds), list(y_test)))
            print('\n')
            print('Classification Results:')
            print(logreg_1_results)
            plot_confusion_matrix(logreg_1_confusion_matrix, classes,
                                  normalize=False,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Purples)
            filename = './models/logreg_optimized.sav'
            pickle.dump(logreg_1, open(filename, 'wb'))
            print('Optimzed Model Saved')        

