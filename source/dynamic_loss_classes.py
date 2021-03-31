"""
dynamic_loss_classes.py: Function to compute the classes, which can be weighted
differently in the loss function for training the NN.
i.e. we divide the training window w.r.t theta (theta_min, theta_max)
into m classes s.t. the variability of the optimal coefficients minising the
variance is the same in each class, which means:
    sum_i (coeff_Fu(num_sample, theta_k) - coeff_Fu(num_sample, theta_k+1))
is the same for each k=1,...,m

Created on Fri Mar  5 15:15:22 2021

@author: Franz Baumdicker, Klara Burger
"""

import numpy
import sys_path
from source.base.opt_coefficients import coeff_Fu


def loss_class(num_samples, num_class, theta_min, theta_max, tol):
    # check if inputs are valid
    if isinstance(num_class, (int, numpy.integer)) == 0:
        return print("ERROR: the number of classes has to be an integer!")
    if theta_min > theta_max:
        return print("ERROR: theta_min has to be smaller than theta_max!")
    if num_class < 1:
        return print("ERROR: there has to be atleast one class!")

    # vector definining the classes
    classes = numpy.zeros((num_class + 1,))

    # set classes boundaries to minimal & maximal theta in the training data
    classes[0] = theta_min
    classes[num_class] = theta_max

    # if there should be only one class to weight the loss, return vector classes
    if num_class == 1:
        return classes

    # if theta<1 set upper boundary of the first class to 1 (since the change
    # in the MVUE coeff. are the largest for small theta, i.e. theta<1)
    if theta_min < 1:
        classes[1] = 1  # set upper boundary of first class to 1

        # compute the total change in the coefficients of MVUE for the range of
        # possible theta values.
        sum1 = 0
        for i in range(0, num_samples - 1):
            sum1 = (
                sum1
                + coeff_Fu(num_samples, 1)[0, i]
                - coeff_Fu(num_samples, theta_max)[0, i]
            )
        total_var = sum1
        # compute the average change in coefficients per class
        var_class = total_var / (num_class - 1)
        # set the tolerance to what extend the change in coeff. per class can
        # distinguish from the average change
        eps = tol * var_class

        for k in range(2, num_class):
            theta_test = classes[k - 1]
            sum2 = 0
            while abs(sum2 - var_class) > eps:
                sum2 = 0
                theta_test = numpy.round(theta_test + 0.1, 1)
                for i in range(0, num_samples - 1):
                    sum2 = (
                        sum2
                        + coeff_Fu(num_samples, classes[k - 1])[0, i]
                        - coeff_Fu(num_samples, theta_test)[0, i]
                    )
                classes[k] = numpy.round(theta_test, 1)

    else:
        # compute the total change in the coefficients of MVUE for the range of
        # possible theta values.
        sum1 = 0
        for i in range(0, num_samples - 1):
            sum1 = (
                sum1
                + coeff_Fu(num_samples, theta_min)[0, i]
                - coeff_Fu(num_samples, theta_max)[0, i]
            )
        total_var = sum1
        # compute the average change in coefficients per clas
        var_class = total_var / (num_class)
        # set the tolerance to what extend the change in coeff. per class can
        # distinguish from the average change
        eps = tol * var_class

        for k in range(1, num_class):
            theta_test = classes[k - 1]
            sum2 = 0
            while abs(sum2 - var_class) > eps:
                sum2 = 0
                theta_test = numpy.round(theta_test + 0.1, 1)
                for i in range(0, num_samples - 1):
                    sum2 = (
                        sum2
                        + coeff_Fu(num_samples, classes[k - 1])[0, i]
                        - coeff_Fu(num_samples, theta_test)[0, i]
                    )
                classes[k] = numpy.round(theta_test, 1)

    return classes
