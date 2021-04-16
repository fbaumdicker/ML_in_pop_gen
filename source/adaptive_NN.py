#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adaptive_NN.py: Function to train an adaptive dense feedforward NN with
1 hidden layer.
Adaptive: divide the training window in classes w.r.t. theta. Classes are chosen
s.t. the change of the coefficients of the estimator by Fu is the same in each class.
According to the performance of the NN per class in comparison to all model-based
estimators and the linear NN put more weight on classes of comparatively
poor performance. Extent of weight increase depends on deviation from the best
estimator the NN is compared to.

Created on Tue Mar  9 16:23:48 2020

@author: Franz Baumdicker, Klara Burger
"""

from tensorflow import keras
import numpy
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import tensorflow.keras.backend as K


import sys_path
from source.dynamic_loss_classes import loss_class
from source.model_based_estimators import ItV, ItMSE, watterson


def nmse(y_true, y_pred, a, c):
    dim = len(c) - 1
    loss_classes = [[] for i in range(dim)]
    cond = [[] for i in range(dim)]
    loss = [[] for i in range(dim)]

    for i in range(0, dim):
        loss_classes[i] = a[i] * K.square((y_pred - y_true) / (y_true))
        cond[i] = K.less(y_true, c[i + 1]) & K.greater(y_true, c[i])

    loss[0] = K.switch(cond[0], loss_classes[0], loss_classes[1])
    for i in range(1, dim):
        loss[i] = K.switch(cond[i], loss_classes[i], loss[i - 1])

    return K.mean(loss[dim - 1], axis=-1)


def loss_wrapper(a, c):
    def nmse_loss(y_true, y_pred):
        return nmse(y_true, y_pred, a, c)

    return nmse_loss


def normalise_coeff(c, n):
    norm_c = numpy.zeros((n,))
    sum = 0
    for i in range(0, n):
        sum = sum + c[i]
    for i in range(0, n):
        norm_c[i] = numpy.round(c[i] / sum, 2)
    return norm_c


def train_adaptive_NN_1hl(
    n,
    num_hidden_nodes,
    num_class,
    num_NN,
    theta_min,
    theta_max,
    tol,
    max_it,
    sloppiness,
    sim_filepath,
    sim_filename,
    linear_NN_filepath,
    linear_NN_filename,
    save_filepath,
):

    # check if input is reasonable

    # load training data
    sim_data = sim_filepath + sim_filename + ".npz"
    input_train = numpy.load(sim_data)

    linear_NN = linear_NN_filepath + linear_NN_filename

    print("load data")
    # convert training data into numpy arrays and then into pandas dataframes
    df1 = []
    df2 = []
    df3 = []
    df4 = []
    X_all = []
    Y_all = []
    for SFS, theta in zip(input_train["multi_SFS"], input_train["multi_theta"]):
        df1.append(SFS[0 : n - 1].tolist())
        df2.append(theta)
    df3 = numpy.array(df1)
    df4 = numpy.array(df2)
    X_all = pd.DataFrame(df3)
    Y_all = pd.DataFrame(df4)

    # splitting the training data into train(80%) and validation (20%) data
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_all, Y_all, train_size=0.8, shuffle=True
    )

    X_val = numpy.array(X_valid)
    y_val = numpy.array(y_valid)

    X_train_NN, X_valid_NN, y_train_NN, y_valid_NN = train_test_split(
        X_train, y_train, train_size=0.8, shuffle=True
    )

    print("create classes for adaptive loss function (this step may take a few minutes)")
    # create classes for loss function
    c = loss_class(n, num_class, theta_min, theta_max, tol)

    # divide samples from validation data set into classes
    X_valid_class = [[] for i in range(num_class)]
    y_valid_class = [[] for i in range(num_class)]
    for j in range(0, len(y_valid)):
        for k in range(0, num_class + 1):
            if y_val[j] >= c[k] and y_val[j] < c[k + 1]:
                X_valid_class[k].append(X_val[j, :])
                y_valid_class[k].append(y_val[j][0])

    # compute nmse for all estimators used for comparison
    Wat_est = [[] for i in range(num_class)]
    Wat_nmse = [[] for i in range(num_class)]
    ItV_est = [[] for i in range(num_class)]
    ItV_nmse = [[] for i in range(num_class)]
    ItMSE_est = [[] for i in range(num_class)]
    ItMSE_nmse = [[] for i in range(num_class)]
    Linear_NN_est = [[] for i in range(num_class)]
    Linear_NN_nmse = [[] for i in range(num_class)]

    for k in range(0, num_class):
        ItV_est[k].append(ItV(numpy.array(X_valid_class[k]), 1e-3))
        ItMSE_est[k].append(ItMSE(numpy.array(X_valid_class[k]), 1e-3))
        Linear_NN_est[k].append(
            keras.models.load_model(linear_NN, compile=False).predict(
                pd.DataFrame(X_valid_class[k])
            )
        )
        for j in range(len(y_valid_class[k])):
            Wat_est[k].append(watterson(X_valid_class[k][j]))
        ItV_nmse[k].append(
            mean_squared_error(
                numpy.sqrt(y_valid_class[k]),
                numpy.divide(ItV_est[k][0], numpy.sqrt(y_valid_class[k]).reshape(-1)),
            )
        )
        ItMSE_nmse[k].append(
            mean_squared_error(
                numpy.sqrt(y_valid_class[k]),
                numpy.divide(ItMSE_est[k][0], numpy.sqrt(y_valid_class[k]).reshape(-1)),
            )
        )
        Wat_nmse[k].append(
            mean_squared_error(
                numpy.sqrt(y_valid_class[k]),
                numpy.divide(Wat_est[k], numpy.sqrt(y_valid_class[k]).reshape(-1)),
            )
        )
        Linear_NN_nmse[k].append(
            mean_squared_error(
                numpy.sqrt(y_valid_class[k]),
                numpy.divide(
                    Linear_NN_est[k][0].reshape(-1),
                    numpy.sqrt(y_valid_class[k]).reshape(-1),
                ),
            )
        )

    coefficients = [[] for j in range(num_NN)]
    coeff_normalised = [[] for j in range(num_NN)]
    it = numpy.zeros((num_NN,))

    print("start training NN adaptively")
    for j in range(1, num_NN + 1):
        # reset weights for NN
        model1 = keras.models.Sequential(
            [
                keras.layers.Dense(
                    num_hidden_nodes,
                    activation="relu",
                    use_bias=True,
                    input_shape=(n - 1,),
                ),
                keras.layers.Dense(1, use_bias=False),
            ]
        )
        reset_weights = model1.get_weights()
        a = numpy.ones((num_class,))

        for i in range(0, max_it):
            check = 0
            NN = keras.models.Sequential(
                [
                    keras.layers.Dense(
                        num_hidden_nodes,
                        activation="relu",
                        use_bias=True,
                        input_shape=(n - 1,),
                    ),
                    keras.layers.Dense(1, use_bias=False),
                ]
            )
            if i == 0:
                NN.set_weights(reset_weights)

            custom_loss = loss_wrapper(a, c)
            NN.compile(loss=custom_loss, optimizer="adam")

            callbacks = [
                EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)
            ]

            NN.fit(
                X_train_NN,
                y_train_NN,
                epochs=500,
                callbacks=callbacks,
                validation_data=(X_valid_NN, y_valid_NN),
                batch_size=64,
            )

            NN_est = [[] for j in range(num_class)]
            NN_nmse = [[] for j in range(num_class)]

            for k in range(0, num_class):
                NN_est[k].append(NN.predict(pd.DataFrame(X_valid_class[k])))
                NN_nmse[k].append(
                    mean_squared_error(
                        numpy.sqrt(y_valid_class[k]),
                        numpy.divide(
                            NN_est[k][0].reshape(-1),
                            numpy.sqrt(y_valid_class[k]).reshape(-1),
                        ),
                    )
                )

            benchmark_min = [[] for j in range(num_class)]
            D = numpy.zeros((num_class,))
            NMSE_p = numpy.zeros((num_class,))
            for k in range(0, num_class):
                benchmark_min[k] = numpy.amin(
                    [
                        Wat_nmse[k][0],
                        ItV_nmse[k][0],
                        ItV_nmse[k][0],
                        Linear_NN_nmse[k][0],
                    ]
                )
                D[k] = NN_nmse[k][0] - benchmark_min[k]
                NMSE_p[k] = numpy.maximum(D[k] / benchmark_min[k], 0)

            M = numpy.amax(NMSE_p)
            if M == 0:
                print("iteration:", i, "M=0")
                break

            for k in range(0, num_class):
                if NN_nmse[k] > (1 + sloppiness) * benchmark_min[k]:
                    a[k] = a[k] + random.uniform(0.25, 0.5) * NMSE_p[k] / M
                    check = check + 1

            if check == 0:
                break
            print("iteration:", i)

            # if maximal number of epoches has been reached, signal it in the
            # following training statistics
            if i == max_it - 1:
                print(
                    "ERROR: maximal number of iterations for training has been reached",
                    j,
                    "! Start training again.",
                )
                it[j - 1] = -1
                coefficients[j - 1].append(numpy.zeros((num_class,)))
                coeff_normalised[j - 1].append(numpy.zeros((num_class,)))
                print("coefficients:", coefficients)
                print("normalised coefficients:", coeff_normalised)
                print("iterations:", it)

        # if training completed properly, safe training statistics and NN
        if check == 0 or M == 0:
            print("number of iterations for training:", i + 1, "NN Nr.:", j)
            it[j - 1] = i + 1
            coefficients[j - 1].append(numpy.round(a, 2))
            coeff_normalised[j - 1].append(
                normalise_coeff(numpy.round(a, 2), num_class)
            )
            print("coefficients:", coefficients)
            print("normalised coefficients:", coeff_normalised)
            print("iterations:", it)
            # save NN
            filename = (
                save_filepath
                + "adaptive_NN_1hl_"
                + str(num_hidden_nodes)
                + "_"
                + sim_filename
                + "_"
                + str(j)
            )
            NN.save(filename)
            print("NN", j, "has been trained SUCCESSFULLY!")
            print("NN", j, "has been saved in:", filename)
