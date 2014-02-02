#!/usr/bin/env python
from pylearn2.models import mlp
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.training_algorithms import sgd, learning_rule
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets import DenseDesignMatrix
from pylearn2.train import Train
from pylearn2.train_extensions import best_params
from pylearn2.space import VectorSpace
import pickle
import numpy as np

def to_one_hot(l):
    out = np.zeros((len(l), len(set(l))))
    for n, i in enumerate(l):
        out[n, i] = 1.
    return out

x = pickle.load(open('saved_x.pkl', 'rb'))
y = pickle.load(open('saved_y.pkl', 'rb'))
y = to_one_hot(y)
in_space = VectorSpace(dim=x.shape[1])
full = DenseDesignMatrix(X=x, y=y)

l1 = mlp.RectifiedLinear(layer_name='l1',
                         sparse_init=12,
                         dim=5000,
                         max_col_norm=1.)

l2 = mlp.RectifiedLinear(layer_name='l2',
                         sparse_init=12,
                         dim=5000,
                         max_col_norm=1.)

l3 = mlp.RectifiedLinear(layer_name='l3',
                         sparse_init=12,
                         dim=5000,
                         max_col_norm=1.)

l4 = mlp.RectifiedLinear(layer_name='l4',
                         sparse_init=12,
                         dim=5000,
                         max_col_norm=1.)

output = mlp.HingeLoss(layer_name='y',
                       n_classes=2,
                       irange=.0001)

#output = mlp.Softmax(layer_name='y',
#                     n_classes=2,
#                     irange=.005)

layers = [l1, l2, l3, output]

mdl = mlp.MLP(layers,
              input_space=in_space)

lr = .0001
epochs = 100
trainer = sgd.SGD(learning_rate=lr,
                  batch_size=128,
                  learning_rule=learning_rule.Momentum(.5),
                  # Remember, default dropout is .5
                  cost=Dropout(input_include_probs={'l1': .8},
                               input_scales={'l1': 1.}),
                  termination_criterion=EpochCounter(epochs),
                  monitoring_dataset={'train': full})

watcher = best_params.MonitorBasedSaveBest(
    channel_name='train_y_misclass',
    save_path='saved_clf.pkl')

velocity = learning_rule.MomentumAdjustor(final_momentum=.6,
                                          start=1,
                                          saturate=250)

decay = sgd.LinearDecayOverEpoch(start=1,
                                 saturate=250,
                                 decay_factor=lr*.05)

experiment = Train(dataset=full,
                   model=mdl,
                   algorithm=trainer,
                   extensions=[watcher, velocity, decay])

if __name__ == "__main__":
    experiment.main_loop()
