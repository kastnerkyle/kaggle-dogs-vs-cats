#!/bin/bash
rm log.log
rm tmp.log
for i in `seq 1 3`; do
    ./kaggle_train_full.py 2>&1 | tee -a log.log
    mv saved_clf.pkl ensemble_clf/clf$i.pkl
    echo "FINISHED RUN $i" >> tmp.log
done
