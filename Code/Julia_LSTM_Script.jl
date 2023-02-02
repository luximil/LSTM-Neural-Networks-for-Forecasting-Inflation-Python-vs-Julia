# This Julia script was written by Francisco Jose Manjon Cabeza Garcia (the author) in the Summer semester
# 2022 for the TU Berlin seminar "Recent Advances in Computational Macroeconomics". The script is a
# practical implementation of a benchmark model, which the author used to compare programming languages for
# researchers. The topic of the seminar paper was based on the paper called "MatLab, Python, Julia: What to
# Choose in Economics?" by Chase Coleman, Spencer Lyon, Lilia Maliar and Serguei Maliar. A presentation of
# this paper was delivered for the seminar, and the paper written by the author shows a small contribution
# to the research direction of the original paper mentioned.
#
# This program is structured in 6 blocks:
# 1. User parameters:
#    Here the user can modify the most important parameters to run the script.
# 2. Import packages and methods:
#    Here the necessary external packages and methods are imported.
# 3. Import and prepare raw data:
#    Here the input data, which also contains the target variable data, is read.
# 4. Define a method to get the neural network model:
#    Here two methods are defined to generate the untrained benchmark model and to train it according to
#    the specifications.
# 5. Create a model and train it at every retraining point:
#    Here the model training and prediction's computations are performed.
# 6. Plot prediction results and observed target value:
#    Here the prediction vs observed plot is generated and prediction accuracy measures are computed.

# ---------------------------------------------------------------------------------------------------------
# User parameters.                                                                                        |
# ---------------------------------------------------------------------------------------------------------

data_csv_path = "C:/Users/labelname/Documents/neural-networks-computational-macroeconomics/Final_data_2022.09.26.csv"
best_param_path = "C:/Users/labelname/Documents/neural-networks-computational-macroeconomics/best_params.bson"
observed_vs_predicted_plot_path = "C:/Users/labelname/Documents/neural-networks-computational-macroeconomics/Plots/"
h = 3                       # Number of months in advance to forecast.
lags = 48                   # Number of past periods to feed into the LSTM cells.
lstm_memory_units = 2       # Number of LSTM cells in the first hidden layer of the network.
dense_layers = 4            # Number of dense layers after the LSTM layer.
dense_layers_units = 128    # Number of cells in each of the hidden dense layers of the network.
val_data_split = 0.25       # Percentage of available data to use for validation at each training point.
warmup_periods = 192        # Number of periods to first train the model.

# ---------------------------------------------------------------------------------------------------------
# Import packages and methods.                                                                            |
# ---------------------------------------------------------------------------------------------------------

using DataFrames            # To handle the input data.
using CSV                   # To read the input data.
using Flux                  # ML package.
using Dates                 # To get attributes of Datetime objects.
using BSON: @save, @load    # To save and load model parameters while training.
using CairoMakie            # Plotting package.

# ---------------------------------------------------------------------------------------------------------
# Import and prepare raw data.                                                                            |
# ---------------------------------------------------------------------------------------------------------

# Import the data we will use to train the model and make predictions.
all_data = CSV.read(data_csv_path,
                    DataFrame,
                    delim=";",
                    header=1)
# Remove the observations from the last h months, because we do not have labels for them yet.
data = all_data[1:end-3, :]

# ---------------------------------------------------------------------------------------------------------
# Define a method to get the neural network model.                                                        |
# ---------------------------------------------------------------------------------------------------------

function Get_Model(;input_size=128, lstm_memory_units=2, dense_layers=4, dense_layers_units=128)
    #=If no parameters are given, this method returns a model as Julia function object with the
    hyperparameters and model specifications of the optimal LSTM-pool model defined in the paper Predicting
    Inflation with Neural Networks by Livia Paranhos.

    input_size=128          : Number of model features, i.e., data timeseries count
    lstm_memory_units=2     : f_{t|L}-size p
    dense_layers=4          : Layers Q
    dense_layers_units=128  : Nodes n

    LSTM activation function: hyperbolic tangent
    FF-layer activation function: rectified linear unit
    Last FF-layer activation function: linear
    Kernel initialiser: Glorot Uniform
    Bias initialiser: Zeros
    =#

    # For the LSTM layer, init only affects the weights' matrix, not the bias vector.
    lstm_layer = LSTM(input_size => lstm_memory_units,
                      init=Flux.glorot_uniform,
                      initb=Flux.zeros32)
    layers = Vector{Any}([lstm_layer, x -> x[:,end]])
    previous_layer_output_units = lstm_memory_units

    for _ in 1:dense_layers
        # For Dense layers, init only affects the weights' matrix, not the bias vector.
        # The bias vector is initialised to zeros by default.
        push!(layers, Dense(previous_layer_output_units => dense_layers_units,
                            relu,
                            bias=true,
                            init=Flux.glorot_uniform))
        previous_layer_output_units = dense_layers_units
    end

    # The Dense layer activation function is linear by default.
    # For Dense layers, init only affects the weights' matrix, not the bias vector.
    # The bias vector is initialised to zeros by default.
    push!(layers, Dense(previous_layer_output_units => 1,
                        bias=true,
                        init=Flux.glorot_uniform))
    model = Chain(layers)

    return model
end

function train_with_custom_callbacks!(;loss, model, train_data, val_data, opt, epochs, es_patience, es_min_dist)
    #=This method trains a given model with two custom callbacks: one to save the model parameters if the
    validation loss improves, and another one to early stop training if the validation loss does not improve by
    a certain minimum for some consecutive epochs.
    
    loss            : Flux Loss function with respect to which the model is trained
    model           : Flux model whose parameters will be trained
    train_data      : Vector of tuples (input, label) to use as training data
    val_data        : Vector of tuples (input, label) to use as validation data
    opt             : Flux Optimiser to train the model with
    epochs          : Number of epochs for training
    es_patience     : Maximum epochs after which training is stopped if the validation loss does not improve
    es_min_dist     : Minimum absolute improvement in validation loss

    =#
    
    # Training_loss is declared local so it will be available for logging outside the gradient calculation.
    local validation_loss

    # Define a variable to track the best validation loss on each epoch. Initialise it to the infinity.
    best_validation_loss = Inf
    # Define a variable to track the number of epochs without validation loss improvement.
    epochs_without_validation_loss_improvement = 0

    # Get the model parameters which are going to be optimised/trained.
    ps = Flux.params(model)

    for _ in 1:epochs
        # Train the model.
        Flux.Optimise.train!(loss, ps, train_data, opt)
        
        # Define two vectors to save the validation predictions and labels.
        validation_predictions = Vector{Float32}([])
        validation_labels = Vector{Float32}([])
    
        for tuple in val_data
            # Compute the prediction for each validation item.
            push!(validation_predictions, model(tuple[1])[1])
            # Save the corresponding label.
            push!(validation_labels, tuple[2])
        end
        
        # Compute validation loss after training.
        validation_loss = loss(validation_predictions, validation_labels)

        # If the new validation loss is not better than the previous validation loss by at least
        # es_min_dist, count this epoch as one without validation loss improvement.
        if best_validation_loss - validation_loss < es_min_dist
            epochs_without_validation_loss_improvement += 1 
        
        # Else, reset counter for epochs without validation loss improvement, and save the current
        # model parameters.
        else
            @save best_param_path model
            epochs_without_validation_loss_improvement = 0
            best_validation_loss = validation_loss
        end

        # If the number of consecutive epochs without validation loss improvement is greater or equal
        # to the early stopping patience, stop training.
        if epochs_without_validation_loss_improvement >= es_patience
            break 
        end
    end
end

# ---------------------------------------------------------------------------------------------------------
# Create a model and train it at every retraining point.                                                  |
# ---------------------------------------------------------------------------------------------------------

# Define a vector to save the model predictions into.
predictions = Vector{Float32}([])
# Define a global model variable.
model = nothing
# Define a global variable for the normalisation factors.
normalisation_factors = nothing

for i in warmup_periods:size(data)[1]
    if Dates.month(data[i,1]) == 12 || model === nothing
        # Either a new year worth of new data is available or this is the start of the loop.
        # Retrain model.
        println(string(data[i,1]) * " - Retraining...")

        # Compute normalisation factors for each feature, so that their values lie in [-1,1].
        # We do not use the data from the last h periods, because we do not have labels for them.
        global normalisation_factors = maximum(abs.(Matrix{Float32}(data[1:i-h, 2:end])'), dims=2)
        
        # Compute the number of time steps to use as training sample. The rest is used as validation sample.
        training_elements = trunc(Int, (i-h) * (1-val_data_split))
        # Create a training dataset with training_elements-many elements.
        training_dataset = Vector{Any}([])
        # Create a validation dataset with (total-training_elements)-many elements.
        validation_dataset = Vector{Any}([])
        
        for j in 1:i-lags-h+1
            # A potential source of ambiguity with RNN in Flux can come from the different data layout
            # compared to some common frameworks where data is typically a 3 dimensional array of the form
            # (features, seq length, samples). In Flux, those 3 dimensions are provided through a vector
            # of seq length containing a matrix (features, samples).
            if j < training_elements
                push!(training_dataset, (Matrix{Float32}(data[j:j+lags-1, 2:end])' ./ normalisation_factors,
                                         data[j+lags+h-1, :CPIAUCNS])) # input data in format (features, lags)
            else
                push!(validation_dataset, (Matrix{Float32}(data[j:j+lags-1, 2:end])' ./ normalisation_factors,
                                           data[j+lags+h-1, :CPIAUCNS])) # input data in format (features, lags)
            end
        end
        
        # Create a new untrained model.
        global model = Get_Model(input_size=size(data)[2]-1)
        # Train the new model.
        train_with_custom_callbacks!(loss=Flux.Losses.mse,
                                     model=model,
                                     train_data=training_dataset,
                                     val_data=validation_dataset,
                                     opt=Flux.Optimise.Adam(),
                                     epochs=400,
                                     es_patience=3,
                                     es_min_dist=0.00001)

        # Load the last saved paramaters, which showed the lowest validation loss.
        @load best_param_path model
    end

    # Get prediction for inflation in h periods using all data available until the end of the i-th month.
    println(string(data[i,1]) * " - Predicting...")
    # Get the last lags-many observations of data at point i in time, and normalise it with the normalisation
    # factors computed during the last retraining process.
    prediction_input = Matrix{Float32}(data[i-lags+1:i,2:end])' ./ normalisation_factors
    # Get and save the model prediction for prediction_input.
    push!(predictions, model(prediction_input)[1])
end

# ---------------------------------------------------------------------------------------------------------
# Plot prediction results and observed target value.                                                      |
# ---------------------------------------------------------------------------------------------------------

# Get the minimum and maximum year to label the x-axis in the plot.
minyear = Dates.year(data[warmup_periods+h:end, :date][1])
maxyear = Dates.year(data[warmup_periods+h:end, :date][end])

# Create a new Figure object.
fig = Figure(resolution=(640,480))
# Create a new Axis object with year's labels every 24 months (x-axis ticks).
ax = Axis(fig[1,1], xticks=(1:24:length(data[warmup_periods:end-h, :date]), string.(minyear:2:maxyear)))
# Plot the predicted values.
lines!(ax, predictions[1:end-h], label="Prediction")
# Plot the observed values.
lines!(ax, data[warmup_periods:end-h, :CPIAUCNS], label="Observed")
# Add a legend to the plot in the upper left corner.
axislegend(position=:lt)
# Save the resulting plot.
save(observed_vs_predicted_plot_path * "$(replace(string(now()), ":" => "."))_Julia_Predicted_vs_Observed_Plot.png", fig)

# ---------------------------------------------------------------------------------------------------------
# Print predicted vs observed statistical figures.                                                        |
# ---------------------------------------------------------------------------------------------------------

# Compute the difference between each prediction and its corresponding observation.
prediction_vs_observed_diffs = predictions[1:end-h] - Vector{Float32}(data[warmup_periods:end-h, :CPIAUCNS])
# Compute the mean squared error of the predictions w.r.t. the observations.
prediction_vs_observed_mse = round((1/length(predictions[1:end-h]))*sum(prediction_vs_observed_diffs.^2),
                                   digits=5)
# Print the mean squared error.
println("Mean squared error (prediction vs observed) $prediction_vs_observed_mse.")

# Compute the absolute differences between each prediction and its corresponding observation.
prediction_vs_observed_abs_diffs = abs.(prediction_vs_observed_diffs)
# Compute the mean absolute error of the predictions w.r.t. the observations.
prediction_vs_observed_mae = round((1/length(predictions[1:end-h]))*sum(prediction_vs_observed_abs_diffs),
                                   digits=5)
# Print the mean absolute error.
println("Mean absolute error (prediction vs observed) $prediction_vs_observed_mae.")

# Compute the maximum absolute error of the predictions w.r.t. the observations.
prediction_vs_observed_maxe = round(maximum(prediction_vs_observed_abs_diffs), digits=5)
# Print the maximum absolute error.
println("Maximum absolute error (prediction vs observed) $prediction_vs_observed_maxe.")