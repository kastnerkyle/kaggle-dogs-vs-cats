#!/usr/bin/env python
from pylearn2.models import mlp, maxout
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.training_algorithms import sgd, learning_rule
from pylearn2.termination_criteria import MonitorBased
from kaggle_dataset import kaggle_dogsvscats
from pylearn2.datasets import preprocessing
from pylearn2.space import Conv2DSpace
from pylearn2.train import Train
from pylearn2.train_extensions import best_params
from pylearn2.utils import serial

trn = kaggle_dogsvscats('train',
                        one_hot=True,
                        datapath='/home/kkastner/kaggle_data/kaggle-dogs-vs-cats',
                        axes=('c', 0, 1, 'b'))

tst = kaggle_dogsvscats('valid',
                        one_hot=True,
                        datapath='/home/kkastner/kaggle_data/kaggle-dogs-vs-cats',
                        axes=('c', 0, 1, 'b'))

in_space = Conv2DSpace(shape=(32, 32),
                       num_channels=3,
                       axes=('c', 0, 1, 'b'))

l1 = maxout.MaxoutConvC01B(layer_name='l1',
                           pad=4,
                           tied_b=1,
                           W_lr_scale=.05,
                           b_lr_scale=.05,
                           num_channels=48,
                           num_pieces=2,
                           kernel_shape=(8, 8),
                           pool_shape=(4, 4),
                           pool_stride=(2, 2),
                           irange=.005,
                           max_kernel_norm=.9)

l2 = maxout.MaxoutConvC01B(layer_name='l2',
                           pad=3,
                           tied_b=1,
                           W_lr_scale=.05,
                           b_lr_scale=.05,
                           num_channels=128,
                           num_pieces=2,
                           kernel_shape=(8, 8),
                           pool_shape=(4, 4),
                           pool_stride=(2, 2),
                           irange=.005,
                           max_kernel_norm=1.9365)

l3 = maxout.MaxoutConvC01B(layer_name='l3',
                           pad=3,
                           tied_b=1,
                           W_lr_scale=.05,
                           b_lr_scale=.05,
                           num_channels=128,
                           num_pieces=2,
                           kernel_shape=(5, 5),
                           pool_shape=(2, 2),
                           pool_stride=(2, 2),
                           irange=.005,
                           max_kernel_norm=1.9365)

l4 = maxout.Maxout(layer_name='l4',
                   irange=.005,
                   num_units=240,
                   num_pieces=5,
                   max_col_norm=1.9)

output = mlp.Softmax(layer_name='y',
                     n_classes=2,
                     irange=.005,
                     max_col_norm=1.9365)

layers = [l1, l2, l3, l4, output]

mdl = mlp.MLP(layers,
              input_space=in_space)

trainer = sgd.SGD(learning_rate=.001,
                  batch_size=128,
                  learning_rule=learning_rule.Momentum(.8),
                  # Remember, default dropout is .5
                  cost=Dropout(input_include_probs={'l1': .8},
                               input_scales={'l1': 1.}),
                  termination_criterion=MonitorBased(
                      channel_name='valid_y_misclass',
                      prop_decrease=0.,
                      N=10),
                  monitoring_dataset={'valid': tst,
                                      'train': trn})

preprocessor = preprocessing.ZCA()
trn.apply_preprocessor(preprocessor=preprocessor, can_fit=True)
tst.apply_preprocessor(preprocessor=preprocessor, can_fit=False)
serial.save('kaggle_dogsvscats_preprocessor.pkl', preprocessor)

watcher = best_params.MonitorBasedSaveBest(
    channel_name='valid_y_misclass',
    save_path='kaggle_dogsvscats_maxout_zca.pkl')

velocity = learning_rule.MomentumAdjustor(final_momentum=.9,
                                          start=1,
                                          saturate=250)

decay = sgd.LinearDecayOverEpoch(start=1,
                                 saturate=250,
                                 decay_factor=.01)

experiment = Train(dataset=trn,
                   model=mdl,
                   algorithm=trainer,
                   extensions=[watcher, velocity, decay])

experiment.main_loop()
