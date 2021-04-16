#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NN.py: Functions to train a dense feedforward NN with 1 hidden layer (without
adaptive loss) and a linear dense feedforward NN.
Created on Tue Dec  1 16:23:48 2020

@author: Franz Baumdicker, Klara Burger
"""

from tensorflow import keras
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K


# define own loss function keras_nmse
def keras_nmse(y_true, y_pred):
    loss = K.mean(K.square((y_pred - y_true) / (y_true)), axis=-1)
    return loss


def train_NN_1hl(
    num_samples, num_hidden_nodes, sim_filepath, sim_filename, save_filepath
):
    # reset model weights
    model = keras.models.Sequential(
        [
            keras.layers.Dense(
                num_hidden_nodes,
                activation="relu",
                use_bias=True,
                input_shape=(num_samples - 1,),
            ),
            keras.layers.Dense(1, use_bias=False),
        ]
    )
    reset_weights = model.get_weights()

    # load training data
    sim_data = sim_filepath + sim_filename + ".npz"
    input_train = numpy.load(sim_data)

    # convert training data into numpy arrays and then into pandas dataframes
    df1 = []
    df2 = []
    df3 = []
    df4 = []
    X_all = []
    Y_all = []
    for SFS, theta in zip(input_train["multi_SFS"], input_train["multi_theta"]):
        df1.append(SFS[0 : num_samples - 1].tolist())
        df2.append(theta)
    df3 = numpy.array(df1)
    df4 = numpy.array(df2)
    X_all = pd.DataFrame(df3)
    Y_all = pd.DataFrame(df4)

    # splitting the training data into train(80%) and validation (20%) data
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_all, Y_all, train_size=0.8, shuffle=True
    )

    # define dense NN with 39 input nodes, 1 hidden layer with 50 nodes and one
    # output node. Bias parameter is used within the hidden layer, but set to
    # false in the input/output layer. As activation function ReLU is used.
    NN = keras.models.Sequential(
        [
            keras.layers.Dense(
                num_hidden_nodes,
                activation="relu",
                use_bias=True,
                input_shape=(num_samples - 1,),
            ),
            keras.layers.Dense(1, use_bias=False),
        ]
    )

    # compile the NN with loss function normalised RMSE and Adam
    NN.compile(loss=keras_nmse, optimizer="adam")

    # set callback functions to early stop training, patience is set to 5 and
    # best weights are restored
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    ]

    # reset weights
    NN.set_weights(reset_weights)

    # fit NN with early stopping
    NN.fit(
        X_train,
        y_train,
        epochs=500,
        callbacks=callbacks,
        validation_data=(X_valid, y_valid),
        batch_size=64,
    )

    # save NN
    filename = save_filepath + "NN_1hl_" + str(num_hidden_nodes) + "_" + sim_filename
    NN.save(filename)


def train_linear_NN(num_samples, sim_filepath, sim_filename, save_filepath):
    model_linear1 = keras.models.Sequential(
        [keras.layers.Dense(1, use_bias=False, input_shape=(num_samples - 1,))]
    )
    reset_weights_linear = model_linear1.get_weights()

    # load training data
    sim_data = sim_filepath + sim_filename + ".npz"
    input_train = numpy.load(sim_data)

    # convert training data into numpy arrays and then into pandas dataframes
    df1 = []
    df2 = []
    df3 = []
    df4 = []
    X_all = []
    Y_all = []
    for SFS, theta in zip(input_train["multi_SFS"], input_train["multi_theta"]):
        df1.append(SFS[0 : num_samples - 1].tolist())
        df2.append(theta)
    df3 = numpy.array(df1)
    df4 = numpy.array(df2)
    X_all = pd.DataFrame(df3)
    Y_all = pd.DataFrame(df4)

    # splitting the training data into train(80%) and validation (20%) data
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_all, Y_all, train_size=0.8, shuffle=True
    )

    # define dense NN with 39 input nodes, 1 hidden layer with 50 nodes and one
    # output node. Bias parameter is used within the hidden layer, but set to
    # false in the input/output layer. As activation function ReLU is used.
    # dense NN with no hidden layer, 39 input nodes and 1 output node
    NN_linear = keras.models.Sequential(
        [keras.layers.Dense(1, use_bias=False, input_shape=(num_samples - 1,))]
    )

    # compile the NN with loss function normalised RMSE and Adam
    NN_linear.compile(loss=keras_nmse, optimizer="adam")

    # set callback functions to early stop training, patience is set to 5 and
    # best weights are restored
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    ]

    # reset weihghts
    NN_linear.set_weights(reset_weights_linear)

    # fit linear NN with early stopping
    NN_linear.fit(
        X_train,
        y_train,
        epochs=500,
        callbacks=callbacks,
        validation_data=(X_valid, y_valid),
    )

    # save NN
    filename = save_filepath + "Linear_NN_" + sim_filename
    NN_linear.save(filename)
    print("Linear NN has been trained SUCCESSFULLY!")
    print("Linear NN has been saved in:", filename)
