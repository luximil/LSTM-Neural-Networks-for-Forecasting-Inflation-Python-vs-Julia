# This Python script was written by Francisco Jose Manjon Cabeza Garcia (the author) in the Summer
# semester 2022 for the TU Berlin seminar "Recent Advances in Computational Macroeconomics". The script is
# a practical implementation of a benchmark model, which the author used to compare programming languages
# for researchers. The topic of the seminar paper was based on the paper called "MatLab, Python, Julia:
# What to Choose in Economics?" by Chase Coleman, Spencer Lyon, Lilia Maliar and Serguei Maliar. A
# presentation of this paper was delivered for the seminar, and the paper written by the author shows a
# small contribution to the research direction of the original paper mentioned.
#
# This program is structured in 6 blocks:
# 1. User parameters:
#    Here the user can modify the most important parameters to run the script.
# 2. Import packages and methods:
#    Here the necessary external packages and methods are imported.
# 3. Import and prepare raw data:
#    Here the input data, which also contains the target variable data, is read.
# 4. Define a method to get the neural network model:
#    Here a method is defined to generate and compile the untrained benchmark model as specified.
# 5. Create a model and train it at every retraining point:
#    Here the model training and prediction's computations are performed.
# 6. Plot prediction results and observed target value:
#    Here the prediction vs observed plot is generated and prediction accuracy measures are computed.

# ---------------------------------------------------------------------------------------------------------
# User parameters.                                                                                        |
# ---------------------------------------------------------------------------------------------------------

data_csv_path = "C:/Users/labelname/Documents/neural-networks-computational-macroeconomics/Final_data_2022.09.26.csv"
best_param_path = "C:/Users/labelname/Documents/neural-networks-computational-macroeconomics/best_params/"
observed_vs_predicted_plot_path = "C:/Users/labelname/Documents/neural-networks-computational-macroeconomics/Plots/"
h = 3                       # Number of months in advance to forecast.
lags = 48                   # Number of past periods to feed into the LSTM cells.
lstm_memory_units = 2       # Number of LSTM cells in the first hidden layer of the network.
dense_layers = 4            # Number of dense layers after the LSTM layer.
dense_layers_units = 128    # Number of cells in each of the hidden dense layers of the network.
val_data_split = 0.25       # Percentage of available data to use for validation at each training point.
warmup_periods = 191        # Number of periods to first train the model.

# ---------------------------------------------------------------------------------------------------------
# Import packages and methods.                                                                            |
# ---------------------------------------------------------------------------------------------------------

import numpy as np                                  # To handle data.
import pandas as pd                                 # To handle data.

from tensorflow import keras as K                   # ML modelling package.
from tensorflow.keras.layers import LSTM, Dense     # Neural network layers.


from dateutil.relativedelta import relativedelta    # This method allows to add months to datetimes easily.
from datetime import datetime
import matplotlib.pyplot as plt                     # Plotting package.

# ---------------------------------------------------------------------------------------------------------
# Import and prepare raw data.                                                                            |
# ---------------------------------------------------------------------------------------------------------

# Import the data we will use to train the model and make predictions.
all_data = pd.read_csv(data_csv_path,
                       sep=";",
                       header=0,
                       index_col=0,
                       parse_dates=True)
# Remove the observations from the last h months, because we do not have labels for them yet.
data = all_data.iloc[:-h, :]

# ---------------------------------------------------------------------------------------------------------
# Define a method to get the neural network model.                                                        |
# ---------------------------------------------------------------------------------------------------------

def Get_Model(input_size=128, lags=48, lstm_memory_units=2, dense_layers=4, dense_layers_units=128):
    """If no parameters are given, this method returns a compiled but not yet trained Keras Model object
    with the hyperparameters and model specifications of the optimal LSTM-pool model defined in the paper
    Predicting Inflation with Neural Networks by Livia Paranhos.

    input_size=128          : Number of model features, i.e., data timeseries count
    lags=48                 : Lags
    lstm_memory_units=2     : f_{t|L}-size p
    dense_layers=4          : Layers Q
    dense_layers_units=128  : Nodes n

    LSTM activation function: hyperbolic tangent
    FF-layer activation function: rectified linear unit
    Last FF-layer activation function: linear
    Kernel initialiser: Glorot Uniform
    Bias initialiser: Zeros
    Optimiser: Adam
    Loss function for optimisation: Mean squared error
    """
    inputlayer = K.Input((lags, input_size))
    hiddenlayer = LSTM(lstm_memory_units,
                       activation="tanh",
                       return_sequences=False,
                       kernel_initializer=K.initializers.GlorotUniform(),
                       bias_initializer=K.initializers.Zeros(),
                       name="lstm-layer-1")(inputlayer)
    for i in range(dense_layers):
        hiddenlayer = Dense(dense_layers_units,
                            activation="relu",
                            kernel_initializer=K.initializers.GlorotUniform(),
                            bias_initializer=K.initializers.Zeros(),
                            name="dense-layer"+str(i+1))(hiddenlayer)
    outputlayer = Dense(1,
                        activation="linear",
                        kernel_initializer=K.initializers.GlorotUniform(),
                        bias_initializer=K.initializers.Zeros(),
                        name="output-layer")(hiddenlayer)
    
    model = K.Model(inputs=inputlayer, outputs=outputlayer)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss=K.losses.MeanSquaredError())

    return model

# ---------------------------------------------------------------------------------------------------------
# Create a model and train it at every retraining point.                                                  |
# ---------------------------------------------------------------------------------------------------------

# Create a dictionary to save the predictions index by time.
predictions = dict()
# Define a variable for the normalisation factors of each feature.
normalisation_factors = None
# Define a variable for the LSTM-FF-NN.
model = None

for i in range(warmup_periods, data.shape[0]):
    if data.index[i].month == 12 or model is None:
        # Either a new year worth of new data is available or this is the start of the loop.
        # Retrain model.
        print("{0} - Retraining model...".format(data.index[i]))

        # Get the data available until the i-th index.
        # Do not use the last h data points, because at point i, we do not have the labels for them.
        available_data = data.iloc[:i+1-h,:]
        # Compute normalisation factors for each feature, so that their values lie in [-1,1].
        normalisation_factors = available_data.abs().max(axis=0)
        # Normalise available data.
        available_data = available_data / normalisation_factors

        # Get available labels until the i-th index.
        available_labels = data.loc[:, "CPIAUCNS"].iloc[:i+1]
        # Shift labels by h.
        available_labels = available_labels.shift(-h).iloc[:-h]
        
        # Compute the number of time steps to use as training sample. Use the rest as validation sample.
        training_elements = int(available_data.shape[0] * (1-val_data_split))
        # Create a tensorflow.data.Dataset object from training_elements many observations and labels by
        # creating sequences of length lags.
        training_dataset = K.utils.timeseries_dataset_from_array(data=available_data[:training_elements],
                                                                 targets=available_labels[:training_elements],
                                                                 sequence_length=lags,
                                                                 sequence_stride=1,
                                                                 sampling_rate=1,
                                                                 # All data in one batch as in Livia Paranhos.
                                                                 batch_size=999999999,
                                                                 shuffle=False)
        
        # Create a tensorflow.data.Dataset object from (total_elements - training_elements)-many observations
        # and labels by creating sequences of length lags.
        validation_dataset = K.utils.timeseries_dataset_from_array(data=available_data[training_elements:],
                                                                   targets=available_labels[training_elements:],
                                                                   sequence_length=lags,
                                                                   sequence_stride=1,
                                                                   sampling_rate=1,
                                                                   # All data in one batch as in Livia Paranhos.
                                                                   batch_size=999999999,
                                                                   shuffle=False)
        
        # Create a new untrained model.
        model = Get_Model(data.shape[1],
                          lags=lags,
                          lstm_memory_units=lstm_memory_units,
                          dense_layers=dense_layers,
                          dense_layers_units=dense_layers_units)

        # Train the new model.
        # The model is trained with 2 callbacks: one to save the model parameters if the validation loss
        # improves, and another one to early stop training if the validation loss does not improve by at
        # least 0.001 for 3 consecutive epochs.
        history = model.fit(training_dataset,
                            validation_data=validation_dataset,
                            epochs=400, # As in Livia Paranhos.
                            verbose=False,
                            callbacks=[K.callbacks.ModelCheckpoint(filepath=best_param_path,
                                                                   monitor="val_loss",
                                                                   save_best_only=True,
                                                                   save_weights_only=True,
                                                                   verbose=False),
                                       K.callbacks.EarlyStopping(monitor="val_loss",
                                                                 min_delta=0.00001,
                                                                 patience=3,
                                                                 verbose=False)])
        # Load the last saved parameters.
        # The model parameters will correspond to those with the lowest validation loss.
        model.load_weights(best_param_path)
    
    # Get prediction for inflation in h periods using all data available until the end of the i-th month.
    print("{0} - Predicting...".format(data.index[i]))
    # Get the last lags-many observations of data at point i in time, and normalise it with the normalisation
    # factors computed during the last retraining process.
    prediction_input = data.iloc[i-lags+1:i+1, :] / normalisation_factors
    # Add a new axis so that the input model matches the dimensions (batch_size, timesteps, features).
    prediction_input = prediction_input.to_numpy()[np.newaxis,...]
    # Get the model prediction for prediction_input.
    prediction = model.predict(prediction_input, verbose=False)
    # Get the date of the prediction, i.e., h months from the current i-th point in time.
    prediction_date = data.index[i] + relativedelta(months=h)
    # Save the prediction in the predictions dictionary with the date of the prediction as key.
    # Reshape so that the Numpy array has dimensions (1,). The model returns a Numpy array of dimensions (1,1).
    predictions[prediction_date] = prediction.reshape((1,))

# Create a pandas DataFrame with the predictions and their dates as index.
predictionsdf = pd.DataFrame.from_dict(predictions, orient="index")

# ---------------------------------------------------------------------------------------------------------
# Plot prediction results and observed target value.                                                      |
# ---------------------------------------------------------------------------------------------------------

# Get the plot labels for the x-axis ticks.
xlabels = predictionsdf.index[:-h]
# Get the timeseries of the observations.
observed = all_data.loc[:, "CPIAUCNS"].shift(-h).loc[predictionsdf.index[:-h]]

# Plot the predicted values.
plt.plot(xlabels, predictionsdf.iloc[:-h,0], label="Prediction")
# Plot the observed values.
plt.plot(xlabels, observed, label="Observed")
# Add a legend to the plot in the upper left corner.
plt.legend(loc="upper left")
# Tighten the layout of the plot.
plt.tight_layout()
# Save the resulting plot.
plt.savefig(observed_vs_predicted_plot_path+str(datetime.now()).replace(":", ".")+"_Python_Predicted_vs_Observed_Plot.png")

# ---------------------------------------------------------------------------------------------------------
# Print predicted vs observed statistical figures.                                                        |
# ---------------------------------------------------------------------------------------------------------

# Compute the difference between each prediction and its corresponding observation.
prediction_vs_observed_diffs = predictionsdf.iloc[:-h,0] - observed
# Compute the mean squared error of the predictions w.r.t. the observations.
prediction_vs_observed_mse = np.round(np.mean(np.power(prediction_vs_observed_diffs, 2)), 5)
# Print the mean squared error.
print("Mean squared error (prediction vs observed) {0}.".format(prediction_vs_observed_mse))

# Compute the absolute differences between each prediction and its corresponding observation.
prediction_vs_observed_abs_diffs = np.abs(prediction_vs_observed_diffs)
# Compute the mean absolute error of the predictions w.r.t. the observations.
prediction_vs_observed_mae = np.round(np.mean(prediction_vs_observed_abs_diffs), 5)
# Print the mean absolute error.
print("Mean absolute error (prediction vs observed) {0}.".format(prediction_vs_observed_mae))

# Compute the maximum absolute error of the predictions w.r.t. the observations.
prediction_vs_observed_maxe = np.round(np.max(prediction_vs_observed_abs_diffs), 5)
# Print the maximum absolute error.
print("Maximum absolute error (prediction vs observed) {0}".format(prediction_vs_observed_maxe))