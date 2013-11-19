#!/usr/bin/env python
import pickle
from sklearn.svm import SVC
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import accuracy_score

x = pickle.load(open('saved_x.pkl', 'rb'))
y = pickle.load(open('saved_y.pkl', 'rb'))
X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=.2,
                                                    random_state=42)
n_iter_search = 50
param_dist = {'degree': [1, 2, 3, 4, 5],
              'C': [np.random.randint(1, 4) * np.random.random_sample()
                    for i in range(20)],
              'shrinking': [True, False]}
#Last chosen params were
#{'kernel': 'poly', 'C': 0.6355611416796046, 'verbose': False,
#'probability': False, 'degree': 2, 'shrinking': True, 'max_iter': -1,
#'random_state': None, 'tol': 0.001, 'cache_size': 200, 'coef0': 0.0,
#'gamma': 0.0, 'class_weight': None}

random_search = RandomizedSearchCV(SVC(kernel='poly'),
                                   param_dist,
                                   n_iter=n_iter_search,
                                   refit=True)
random_search.fit(X_train, y_train)
clf = random_search.best_estimator_


def score(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    q = clf.predict(X_test)
    print 'Accuracy', accuracy_score(q, y_test)


print "Scoring raw output"
score(clf, X_train, y_train, X_test, y_test)
print "Now training on full dataset"
clf.fit(x, y)
pickle.dump(clf, open('saved_clf.pkl', 'wb'))
