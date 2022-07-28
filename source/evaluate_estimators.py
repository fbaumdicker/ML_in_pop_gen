#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_estimators.py: script for two functions:
    - evaluate_estimators(): evaluate Watterson, ItMSE, NN and linear NN
      on a given series of data sets (recombination rate fixed and mutation
      rate varies). Results are saved. (For saving computing time, computations
      for ItV, MMSEE, IMVUE have been excluded).
    - evaluate_NN(): evaluate Watterson, ItMSE, NN and linear NN (trained  with
      different recombination rates) on a given series of data sets
      (recombination rate varies and mutation rate is fixed to theta=40).
      Results are saved.

Created on Mon Dec 14 16:55:24 2020

@author: Franz Baumdicker, Klara Burger
"""

# import all needed libraries
from tensorflow import keras
import numpy
import pandas as pd
from sklearn.metrics import mean_squared_error
import sys_path
from source.model_based_estimators import ItMSE, watterson  # MVUE, MMSEE, ItV
from source.save_estimators import save_features


# compare performance Watterson's
# estimator, Minimal Variance Unbiased Estimator (MVUE), Minimal
# MSE Estimator (MMSEE), their iterative versions ItV and ItMS,
# the linear NN and the NN with one hidden layer:
def evaluate_estimators(
    num_sample,
    rep_test_data,
    rho,
    num_hidden_nodes,
    rep_training_data,
    filepath,
    eval_model_based_est,
    save=True,
):
    # mean prediction and normalised MSE will be saved as a vector,
    # in entry i-th the results for test set i:
    Wat_mean = []  # Watterson's estimator
    Wat_nmse = []
    Wat_est = []

    ItMSE_mean = []  # ItMSE
    ItMSE_nmse = []
    ItMSE_est = []

    # ItV_mean = []  # ItV
    # ItV_nmse = []
    # ItV_est = []

    # MMSEE_mean = []  # MMSSEE
    # MMSEE_nmse = []
    # MMSEE_est = []

    # MVUE_mean = []  # MVUE
    # MVUE_nmse = []
    # MVUE_est = []

    NN_mean = []  # NN with one hidden layer
    NN_nmse = []
    NN_est = []

    Linear_NN_mean = []  # Linear NN
    Linear_NN_nmse = []
    Linear_NN_est = []

    # save value of true theta used to simulate SFS, will be used for plotting:
    val = []
    theta_min = 0
    theta_max = 40

    # loop to shift through every test data set respectively,
    # theta = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, ..., 40.
    for i in range(1, 50):
        # clear all dataframes before usage for the next test set
        watterson_est = []
        itmse = []
        # itv = []
        # mmsee = []
        # mvue = []
        y_pred = []
        y_pred_linear = []
        df1 = []
        df2 = []
        df3 = []
        df4 = []
        X_test = []
        y_test = []
        input_test = []
        theta = []

        if i <= 9:
            k = float(i / 10)
        else:
            k = float(i - 9)

        # data handling: load test data, convert into numpy arrays
        # and then into pandas dataframes:

        file = (
            filepath
            + "rho_"
            + str(rho)
            + "/sim_"
            + "n_"
            + str(num_sample)
            + "_rep_"
            + str(rep_test_data)
            + "_rho_"
            + str(rho)
            + "_theta_"
            + str(k)
            + ".npz"
        )

        input_test = numpy.load(file, allow_pickle=True)
        for SFS, theta in zip(
            input_test["multi_SFS"],
            input_test["multi_theta"],
        ):
            df1.append(SFS[0 : num_sample - 1].tolist())
            df2.append(theta)
            watterson_est.append(watterson(SFS))

        df3 = numpy.array(df1)
        df4 = numpy.array(df2)
        X_test = pd.DataFrame(df3)
        y_test = pd.DataFrame(df4)

        # set theta and save value in val for plotting
        theta = y_test.iat[0, 0]
        val.append(theta)

        # NN and linear NN estimation of theta:
        y_pred = keras.models.load_model(
            "../data/saved_NN/adaptive_NN_1hl_200_sim_n_"
            + str(num_sample)
            + "_rep_"
            + str(rep_training_data)
            + "_rho_"
            + str(rho)
            + "_theta_random-100_1",
            compile=False,
        ).predict(X_test)

        y_pred_linear = keras.models.load_model(
            "../data/saved_NN/Linear_NN_sim_n_"
            + str(num_sample)
            + "_rep_"
            + str(rep_training_data)
            + "_rho_"
            + str(rho)
            + "_theta_random-100",
            compile=False,
        ).predict(X_test)

        if eval_model_based_est == 1:
            # print("Compute ItV")
            # itv = ItV(df3, 1e-3)
            # print("Compute ItMSE")
            itmse = ItMSE(df3, 1e-3)
            # print("Compute MMSEE")
            # mmsee = MMSEE(df3, theta)
            # print("Compute MVUE")
            # mvue = MVUE(df3, theta)

            # save characteristics of the estimators predictions in arrays
            Wat_mean.append(numpy.mean(watterson_est))
            Wat_nmse.append(mean_squared_error(y_test, watterson_est) / theta)
            Wat_est.append(watterson_est)

            ItMSE_mean.append(numpy.mean(itmse))
            ItMSE_nmse.append(mean_squared_error(y_test, itmse) / theta)
            ItMSE_est.append(itmse)

            # ItV_mean.append(numpy.mean(itv))
            # ItV_nmse.append(mean_squared_error(y_test, itv) / theta)
            # ItV_est.append(itv)

            # MMSEE_mean.append(numpy.mean(mmsee))
            # MMSEE_nmse.append(mean_squared_error(y_test, mmsee) / theta)
            # MMSEE_est.append(mmsee)

            # MVUE_mean.append(numpy.mean(mvue))
            # MVUE_nmse.append(mean_squared_error(y_test, mvue) / theta)
            # MVUE_est.append(mvue)

        NN_mean.append(numpy.mean(y_pred))
        NN_nmse.append(mean_squared_error(y_test, y_pred) / theta)
        NN_est.append(y_pred)

        Linear_NN_mean.append(numpy.mean(y_pred_linear))
        Linear_NN_nmse.append(mean_squared_error(y_test, y_pred_linear) / theta)
        Linear_NN_est.append(y_pred_linear)

        print("step", i, "of 49 done.")

    # if save is True, save all estimands of estimators and some statistics:
    if save == 1:
        if eval_model_based_est == 1:
            save_features(Wat_mean, "Wat_mean", num_sample, rho, theta_min, theta_max)
            save_features(Wat_nmse, "Wat_nmse", num_sample, rho, theta_min, theta_max)
            save_features(Wat_est, "Wat_est", num_sample, rho, theta_min, theta_max)

            save_features(
                ItMSE_mean,
                "ItMSE_mean",
                num_sample,
                rho,
                theta_min,
                theta_max,
            )
            save_features(
                ItMSE_nmse,
                "ItMSE_nmse",
                num_sample,
                rho,
                theta_min,
                theta_max,
            )
            save_features(ItMSE_est, "ItMSE_est", num_sample, rho, theta_min, theta_max)

            # save_features(
            #        ItV_mean, "ItV_mean", num_sample, rho, theta_min, theta_max
            #        )
            # save_features(
            #        ItV_nmse, "ItV_nmse", num_sample, rho, theta_min, theta_max
            #        )
            # save_features(
            #        ItV_est, "ItV_est", num_sample, rho, theta_min, theta_max
            #        )

            # save_features(
            #        MMSEE_mean,
            #        "MMSEE_mean",
            #        num_sample,
            #        rho,
            #        theta_min,
            #        theta_max,
            #        )
            # save_features(
            #        MMSEE_nmse,
            #        "MMSEE_nmse",
            #        num_sample,
            #        rho,
            #        theta_min,
            #        theta_max,
            #        )
            # save_features(
            #        MMSEE_est, "MMSEE_est", num_sample, rho, theta_min, theta_max
            #        )
            #
            # save_features(
            #        MVUE_mean, "MVUE_mean", num_sample, rho, theta_min, theta_max
            #        )
            # save_features(
            #        MVUE_nmse, "MVUE_nmse", num_sample, rho, theta_min, theta_max
            #        )
            # save_features(
            #        MVUE_est, "MVUE_est", num_sample, rho, theta_min, theta_max
            #        )

        save_features(NN_mean, "NN_mean", num_sample, rho, theta_min, theta_max)
        save_features(NN_nmse, "NN_nmse", num_sample, rho, theta_min, theta_max)
        save_features(NN_est, "NN_est", num_sample, rho, theta_min, theta_max)

        save_features(
            Linear_NN_mean,
            "Linear_NN_mean",
            num_sample,
            rho,
            theta_min,
            theta_max,
        )
        save_features(
            Linear_NN_nmse,
            "Linear_NN_nmse",
            num_sample,
            rho,
            theta_min,
            theta_max,
        )
        save_features(
            Linear_NN_est,
            "Linear_NN_est",
            num_sample,
            rho,
            theta_min,
            theta_max,
        )

        save_features(val, "val", num_sample, rho, theta_min, theta_max)


# compare performance NN, linear NN (trained with different recombination
# rates), Watterson and ItMSE for different recombination rates:
def evaluate_rho_variable(
    n,
    rep_test_data,
    real_theta,
    rho,
    num_hidden_nodes,
    rep_training_data,
    rep_training_data_rho_var,
    theta_min,
    theta_max,
    filepath,
    save=True,
):
    # Normalised MSE will be saved as a vector,
    # in entry i-th the results for test set i:
    NN_rho_0_nmse = []
    NN_linear_rho_0_nmse = []
    NN_rho_35_nmse = []
    NN_linear_rho_35_nmse = []
    NN_rho_1000_nmse = []
    NN_linear_rho_1000_nmse = []
    NN_rho_var_nmse = []
    NN_linear_rho_var_nmse = []
    Wat_nmse = []
    ItMSE_nmse = []

    # loop to shift through every test data set respectively,
    # theta = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, ..., 40.
    # loop to shift through every test dataset with theta=i respectively
    for i in range(0, 52):
        # clear all dataframes before usage for the next testset
        input_test = []
        df1 = []
        df2 = []
        df3 = []
        df4 = []
        X_test = []
        y_test = []
        y_pred_0 = []
        y_pred_linear_0 = []
        y_pred_35 = []
        y_pred_linear_35 = []
        y_pred_1000 = []
        y_pred_linear_1000 = []
        y_pred_var = []
        y_pred_linear_var = []
        watterson_est = []
        ItMSE_est = []
        theta_int = int(real_theta)

        print("real_theta", real_theta, i)

        file_1000 = (
            filepath
            + "/rho_1000/"
            + "sim_n_"
            + str(n)
            + "_rep_"
            + str(rep_test_data)
            + "_rho_1000_theta_"
            + str(real_theta)
            + ".npz"
        )
        file = (
            filepath
            + "/theta_"
            + str(theta_int)
            + "/"
            + "sim_n_"
            + str(n)
            + "_rep_"
            + str(rep_test_data)
            + "_rho_"
            + str(i)
            + "_theta_"
            + str(real_theta)
            + ".npz"
        )
        # load data: the first 50 data sets are simulated with rho=0,1,2,...50,
        # dataset 51 is simulated with rho=1000:
        if i == 51:
            input_test = numpy.load(file_1000)
        else:
            input_test = numpy.load(file)

        # data handling: convert into numpy arrays
        # and then into pandas dataframes:

        # convert data into numpy arrays and then into pandas dataframes
        for SFS, theta in zip(
            input_test["multi_SFS"],
            input_test["multi_theta"],
        ):
            df1.append(SFS[0 : n - 1].tolist())
            df2.append(theta)
            watterson_est.append(watterson(SFS))

        df3 = numpy.array(df1)
        df4 = numpy.array(df2)
        X_test = pd.DataFrame(df3)
        y_test = pd.DataFrame(df4)

        # compute ItMSE
        ItMSE_est = ItMSE(df3, 1e-3)

        # get predictions of the linear NN and the NN with one hidden layer
        y_pred_0 = keras.models.load_model(
            "../data/saved_NN/adaptive_NN_1hl_"
            + str(num_hidden_nodes)
            + "_n_"
            + str(n)
            + "_rep_"
            + str(rep_training_data)
            + "_rho_0_theta_random-100_1",
            compile=False,
        ).predict(X_test)
        y_pred_linear_0 = keras.models.load_model(
            "../data/saved_NN/Linear_NN_n_"
            + str(n)
            + "_rep_"
            + str(rep_training_data)
            + "_rho_0_theta_random-100",
            compile=False,
        ).predict(X_test)
        y_pred_35 = keras.models.load_model(
            "../data/saved_NN/adaptive_NN_1hl_"
            + str(num_hidden_nodes)
            + "_n_"
            + str(n)
            + "_rep_"
            + str(rep_training_data)
            + "_rho_35_theta_random-100_1",
            compile=False,
        ).predict(X_test)
        y_pred_linear_35 = keras.models.load_model(
            "../data/saved_NN/Linear_NN_n_"
            + str(n)
            + "_rep_"
            + str(rep_training_data)
            + "_rho_35_theta_random-100",
            compile=False,
        ).predict(X_test)
        y_pred_1000 = keras.models.load_model(
            "../data/saved_NN/adaptive_NN_1hl_"
            + str(num_hidden_nodes)
            + "_n_"
            + str(n)
            + "_rep_"
            + str(rep_training_data)
            + "_rho_1000_theta_random-100_1",
            compile=False,
        ).predict(X_test)
        y_pred_linear_1000 = keras.models.load_model(
            "../data/saved_NN/Linear_NN_n_"
            + str(n)
            + "_rep_"
            + str(rep_training_data)
            + "_rho_1000_theta_random-100",
            compile=False,
        ).predict(X_test)
        y_pred_var = keras.models.load_model(
            "../data/saved_NN/adaptive_NN_1hl_"
            + str(num_hidden_nodes)
            + "_n_"
            + str(n)
            + "_rep_"
            + str(rep_training_data_rho_var)
            + "_rho_var_theta_random-100_1",
            compile=False,
        ).predict(X_test)
        y_pred_linear_var = keras.models.load_model(
            "../data/saved_NN/Linear_NN_n_"
            + str(n)
            + "_rep_"
            + str(rep_training_data_rho_var)
            + "_rho_var_theta_random-100",
            compile=False,
        ).predict(X_test)

        # save characteristics of the models predictions in arrays
        NN_rho_0_nmse.append(mean_squared_error(y_test, y_pred_0) / real_theta)
        NN_linear_rho_0_nmse.append(
            mean_squared_error(y_test, y_pred_linear_0) / real_theta
        )
        NN_rho_35_nmse.append(mean_squared_error(y_test, y_pred_35) / real_theta)
        NN_linear_rho_35_nmse.append(
            mean_squared_error(y_test, y_pred_linear_35) / real_theta
        )
        NN_rho_1000_nmse.append(mean_squared_error(y_test, y_pred_1000) / real_theta)
        NN_linear_rho_1000_nmse.append(
            mean_squared_error(y_test, y_pred_linear_1000) / real_theta
        )
        NN_rho_var_nmse.append(mean_squared_error(y_test, y_pred_var) / real_theta)
        NN_linear_rho_var_nmse.append(
            mean_squared_error(y_test, y_pred_linear_var) / real_theta
        )
        Wat_nmse.append(mean_squared_error(y_test, watterson_est) / real_theta)
        ItMSE_nmse.append(mean_squared_error(y_test, ItMSE_est) / real_theta)

        # save results in data files:
        rho = -1
        save_features(NN_rho_0_nmse, "NN_rho_0_nmse", n, rho, theta_min, theta_max)
        save_features(NN_rho_35_nmse, "NN_rho_35_nmse", n, rho, theta_min, theta_max)
        save_features(
            NN_rho_1000_nmse,
            "NN_rho_1000_nmse",
            n,
            rho,
            theta_min,
            theta_max,
        )
        save_features(
            NN_rho_var_nmse,
            "NN_rho_var_nmse",
            n,
            rho,
            theta_min,
            theta_max,
        )
        save_features(
            NN_linear_rho_0_nmse,
            "Linear_NN_rho_0_nmse",
            n,
            rho,
            theta_min,
            theta_max,
        )
        save_features(
            NN_linear_rho_35_nmse,
            "Linear_NN_rho_35_nmse",
            n,
            rho,
            theta_min,
            theta_max,
        )
        save_features(
            NN_linear_rho_1000_nmse,
            "Linear_NN_rho_1000_nmse",
            n,
            rho,
            theta_min,
            theta_max,
        )
        save_features(
            NN_linear_rho_var_nmse,
            "Linear_NN_rho_var_nmse",
            n,
            rho,
            theta_min,
            theta_max,
        )
        save_features(ItMSE_nmse, "ItMSE_nmse", n, rho, theta_min, theta_max)
        save_features(Wat_nmse, "Wat_nmse", n, rho, theta_min, theta_max)
