#!/usr/bin/env python
from pylearn2.models import mlp
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.training_algorithms import sgd, learning_rule
from pylearn2.termination_criteria import MonitorBased
from pylearn2.datasets import DenseDesignMatrix
from pylearn2.train import Train
from pylearn2.train_extensions import best_params, window_flip
from pylearn2.space import VectorSpace
import pickle
import numpy as np
from sklearn.cross_validation import train_test_split


def to_one_hot(l):
    out = np.zeros((len(l), len(set(l))))
    for n, i in enumerate(l):
        out[n, i] = 1.
    return out

x = pickle.load(open('saved_x.pkl', 'rb'))
y = pickle.load(open('saved_y.pkl', 'rb'))
y = to_one_hot(y)
X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=.2,
                                                    random_state=42)
in_space = VectorSpace(dim=x.shape[1])
trn = DenseDesignMatrix(X=X_train, y=y_train)
tst = DenseDesignMatrix(X=X_test, y=y_test)

l1 = mlp.RectifiedLinear(layer_name='l1',
                         irange=.001,
                         dim=5000,
                         max_col_norm=1.)

l2 = mlp.RectifiedLinear(layer_name='l2',
                         irange=.001,
                         dim=5000,
                         max_col_norm=1.)

l3 = mlp.RectifiedLinear(layer_name='l3',
                         irange=.001,
                         dim=1000,
                         max_col_norm=1.)

l4 = mlp.RectifiedLinear(layer_name='l4',
                         irange=.001,
                         dim=1000,
                         max_col_norm=1.)

output = mlp.HingeLoss(layer_name='y',
                       irange=.0001)

layers = [l1, l2, l3, l4, output]
layers = [l1, l2, l3, output]
layers = [l1, l2, output]
#layers = [l1, output]

mdl = mlp.MLP(layers,
              input_space=in_space)

lr = .001
epochs = 100
trainer = sgd.SGD(learning_rate=lr,
                  batch_size=128,
                  learning_rule=learning_rule.Momentum(.5),
                  # Remember, default dropout is .5
                  cost=Dropout(input_include_probs={'l1': .8},
                               input_scales={'l1': 1.}),
                  termination_criterion=MonitorBased(
                      channel_name='valid_y_misclass',
                      prop_decrease=0.,
                      N=epochs),
                  monitoring_dataset={'valid': tst,
                                      'train': trn})

watcher = best_params.MonitorBasedSaveBest(
    channel_name='valid_y_misclass',
    save_path='saved_clf.pkl')

velocity = learning_rule.MomentumAdjustor(final_momentum=.98,
                                          start=1,
                                          saturate=100)

decay = sgd.LinearDecayOverEpoch(start=1,
                                 saturate=100,
                                 decay_factor=.05 * lr)

win = window_flip.WindowAndFlipC01B(pad_randomized=8,
                                    window_shape=(32, 32),
                                    randomize=[trn],
                                    center=[tst])
experiment = Train(dataset=trn,
                   model=mdl,
                   algorithm=trainer,
                   extensions=[watcher,decay])

experiment.main_loop()
