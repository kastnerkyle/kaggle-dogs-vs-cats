#!/usr/bin/env python
from kaggle_train_full import *
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=.2,
                                                    random_state=42)
trn = DenseDesignMatrix(X=X_train, y=y_train)
tst = DenseDesignMatrix(X=X_test, y=y_test)
trainer.monitoring_dataset={'valid': tst,
                            'train': trn}
experiment.main_loop()
