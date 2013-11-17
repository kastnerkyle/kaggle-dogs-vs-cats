#!/usr/bin/env python
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

x = pickle.load(open('saved_x.pkl', 'rb'))
y = pickle.load(open('saved_y.pkl', 'rb'))
X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=.2,
                                                    random_state=42)
clf = LogisticRegression()
def score(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    q = clf.predict(X_test)
    print 'Accuracy', accuracy_score(q, y_test)


print "Scoring raw output"
score(clf, X_train, y_train, X_test, y_test)
print "Now training on full dataset"
clf.fit(x, y)
pickle.dump(clf, open('saved_clf.pkl', 'wb'))
