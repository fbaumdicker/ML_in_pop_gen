This directory builds several estimators to predict theta from coalescent simulation und compares them by using the mean squared error.
To perform the results of the paper, several steps must be taken:
1) Simulate data via ms prime (see `scripts/sim_train_data_example.py` and `scripts/sim_test_data_example.py`)
2) Train Neural Networks (see `scripts/train_linear_NN_example.py` and `scripts/train_adaptive_NN_example.py`)
3) Evaluate, both model-based (Watterson, MVUE, MMSEE, ItV, ItMSE) and model-free estimators (linear NN and adaptive NN with one hidden layer) on test data (see `scripts/eval_estimators_example.py`)
4) plot the results (see `scripts/plot_nmse_example.py`, `scripts/plot_bias_example.py`, `scripts/plot_nmse_rho_example.py`)

The directory is organised in three levels:
- source: here you can find all functions to simulate the data, train the NN, compute model-based estimators, evaluate all estimators, save and load results. To reproduce the results from the paper you do not have to adapt anything in this folder
- data: in this folder you can find all simulations, saved NN, saved features (e.g. estimations and normalised MSE of estimators) and optimal coefficients for MVUE and MMSEE. Note, most of these quantities you have to compute first via the scripts in the scripts folder.
- scripts: here you can find all the scripts to call the functions from source. To reproduce the same or similar results to those from the paper, you only have to call scripts in this folder. The scripts to plot the results are hard-coded to match the paper, to customize the plots you will have to modify the 
    plot scripts. 

A short example on how to use the directory:
The following commands will train a linear NN and an adaptiv NN by using precalculated simulation, calculate the model-based estimators (Watterson and ItMSE), test all estimators on test data and plot the results. rho=0 is fixed here, but similarly other results can be obtained. 
1) Unpack needed training data:
    `gzip -d data/simulations/sim_n_40_rep_200000_rho_0_theta_random-100.npz.gz`
2) Unpack needed test data:
    `tar -xf data/simulations/rho_0.tar.gz -C data/simulations/`
3) Change directory:
    `cd scripts`
3) Train linear NN for rho=0: 
    `python3 train_linear_NN_example.py`
4) Train adaptive NN for rho=0 (this step can take a few minutes):
    `python3 train_adaptive_NN_example.py`
5) Evaluate both model-based (Watterson and ItMSE) and model-free estimators (linear NN and adaptive NN with one hidden layer) on test data:
    `python3 eval_estimators_example.py`
6) Plot NMSE for rho=0:
    `python3 plot_nmse_example.py`
    
NOTE: For the plots in the paper, estimators were tested on larger data sets (each test set consists of 10000 SFS instead of 1000 SFS.)


To compute and print the coefficients of Fu's estimator, i.e. the MVUE estimator and Futschik's estimator, i.e. the MMSEE estimator use the command `python3 scripts/compute_numerical_coeff_example.py sample_size true_mutation_rate`.
