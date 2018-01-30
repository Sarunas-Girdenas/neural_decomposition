from neural_decomposition_class import NeuralDecomposition
import pandas as pd

testing_set, training_set = list(), list()
# testing and training sets are lists (could be arrays too)
# that include data we want to analyse

nd = NeuralDecomposition(
    epochs=15,
    forecast_periods=len(testing_set),
    data=training_set,
    L1_reg=0.01,
    units=15,
    batch_size=48,
    validation_length=48)

nd.create_keras_model()

nd.train()

# predict some periods ahead
predictions = nd.predict()

# this function could be used to extract cycles from
# time series
nd.get_cycles()