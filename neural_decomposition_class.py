import numpy as np
from keras.layers import Dense, Input
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import regularizers
from keras.engine.training import Model
from keras.callbacks import ModelCheckpoint
from copy import copy
from tensorflow import sin
from keras.backend import set_value


class NeuralDecomposition:

    def __init__(self, data, optimizer='rmsprop',
                 loss='mean_squared_error', units=10,
                 epochs=300, forecast_periods=100,
                 L1_reg=0.01,
                 validation_length=100,
                 batch_size=32):
        """Purpose: initialise class
        data - time series input
        optimizer - keras optimizer
        loss - keras loss
        """

        self.data = data
        self.n_data = len(self.data)
        self.optimizer = optimizer
        self.loss = loss
        self.units = units
        self.epochs = epochs
        self.forecast_periods = forecast_periods
        self.L1_reg = L1_reg
        self.validation_length = validation_length
        self.batch_size = batch_size

    def create_keras_model(self):
        """Purpose: create keras model
        """

        # one dimensional input data
        input_data = Input(shape=(1 ,), name='input_data')

        # periodical component of the series
        sinusoid = Dense(self.n_data, activation=sin, #np.sin
                         name=None)(input_data)
        # name='sin activation'

        # g(t) function, components are the same as
        # described in the paper: linear, softplus and sigmoid
        linear = Dense(self.units, activation='linear',
                       )(input_data) #name='linear activation', activation='linear'
        softplus = Dense(self.units, activation='softplus',
                         )(input_data) # name='softplus activation'
        sigmoid = Dense(self.units, activation='sigmoid',
                        )(input_data) # name='sigmoid activation'

        # concatenate layers into one
        one_layer = Concatenate()([sinusoid, linear,
                                   softplus, sigmoid])

        # output layer, add L1 regularizer as in the paper
        output_layer = Dense(1, kernel_regularizer=regularizers.l1(self.L1_reg))(one_layer)

        # compile keras model
        keras_model = Model(inputs=[input_data], outputs=[output_layer])
        keras_model.compile(loss=self.loss, optimizer=self.optimizer)

        # initialize weights
        keras_model = NeuralDecomposition.initialize_weights(
            keras_model, self.n_data, self.units)

        # assign to class
        self.keras_model = keras_model

        return None

    @staticmethod
    def initialize_weights(keras_model, n_data, units):
        """Purpose: initialize weights for compiled
        keras model
        n_data - length of the input data
        units - number of units in the network
        """

        # sanity type checking
        if not isinstance(keras_model, Model):
            raise TypeError('Input must be Keras Model!')

        noise = 0.001
        np.random.seed(42)

        # for details about the weight initialization
        # see the paper
        set_value(keras_model.weights[0],
            (2 * np.pi * np.floor(np.arange(n_data) / 2))[np.newaxis, :].astype('float32')) # sin/kernel
        set_value(keras_model.weights[1],
            (np.pi / 2 + np.arange(n_data) % 2 * np.pi / 2).astype('float32')) # sin/bias

        # initialize weights for g(t)
        # we know that there 8 weights we need to initialize
        # g(t) consists of 3 activations: linear, softplus and sigmoid
        # each of them has 2 types: kernel and bias so in total there
        # are 6 things we need to initialize
        # adding/removing activation function would alter this number too
        for layer in range(2, 8):
            if layer == 2:
                set_value(keras_model.weights[layer],
                    (np.ones(shape=(1, units)) + np.random.normal(
                        size=(1, units))*noise).astype('float32')) # linear/kernel
            elif layer in [3, 5, 7]:
                # linear/bias
                # sotfplus/bias
                # sigmoid/bias
                set_value(keras_model.weights[layer],
                    (np.random.normal(size=(units)) * noise).astype('float32'))
            else:
                # softplus/kernel
                # sigmoid/kernel
                set_value(keras_model.weights[layer],
                    (np.random.normal(size=(1, units)) * noise).astype('float32')) # softplus/kernel

        # initialize output layer
        set_value(keras_model.weights[8],
            (np.random.normal(size=(n_data + 3 * units, 1))*noise).astype('float32')) # output/kernel (n+3)
        set_value(keras_model.weights[9],
            (np.random.normal(size=(1)) * noise).astype('float32')) # output/bias

        return keras_model

    def train(self):
        """Purpose: train the model obtained in
        create_keras_model() function
        use scaled input data
        """

        # create x
        x = np.linspace(0, 1, self.n_data)
        self.x = x

        # create y
        y, self.max_min_list= NeuralDecomposition.scale_data(self.data, self.units)
        # sanity check
        assert len(y) == len(self.data)

        # create validation set, last 100 observations
        x_val = x[-self.validation_length:]
        y_val = y[-self.validation_length:]

        x = x[:-self.validation_length]
        y = y[:-self.validation_length]

        # create model checkpoint
        weights_path = 'nd_weights.hdf5'

        model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1,
                                           save_best_only=True, mode='auto')

        callbacks = [model_checkpoint]

        history_model = self.keras_model.fit(
            x, y, epochs=self.epochs, verbose=1,
            batch_size=self.batch_size,
            validation_data=(x_val, y_val),
            callbacks=callbacks)

        self.history_model = history_model

        return None

    def predict(self):
        """Compute predictions, inverse scale the data
        """

        # load weights
        self.keras_model.load_weights('nd_weights.hdf5')

        # compute predictions
        predictions = self.keras_model.predict(
            np.concatenate([self.x, 1 + np.arange(
                1, self.forecast_periods + 1) * self.x[1]])).flatten()

        # assign to class
        # note, that predictions contain predictions for training
        # set and out-of-sample predictions in the same array
        self.predictions = NeuralDecomposition.inverse_scale_data(
            predictions, self.max_min_list, self.units)

        return None

    def get_cycles(self):
        """Purpose: get cyclical component of time series
        Intuition: set the relevant weights to zero and compute
        predictions.
        Model equation: data = cycles + trend
        trend is captured by g(t) function
        """

        # take the trained model
        
        keras_model_cycles = self.keras_model
        
        # these lines are taken from the function initialize_weights()
        # set trend g(t) weights to zero

        for layer in range(2, 8):
            if layer == 2:
                set_value(keras_model_cycles.weights[layer],
                    (np.zeros((1, self.units))).astype('float32')) # linear/kernel
            elif layer in [3, 5, 7]:
                # linear/bias
                # sotfplus/bias
                # sigmoid/bias
                set_value(keras_model_cycles.weights[layer],
                    (np.zeros((self.units))).astype('float32'))
            else:
                # softplus/kernel
                # sigmoid/kernel
                set_value(keras_model_cycles.weights[layer],
                    (np.zeros((1, self.units))).astype('float32')) # softplus/kernel

        # compute cyclical component
        cycles = keras_model_cycles.predict(
            np.concatenate([self.x, 1 + np.arange(
                1, self.forecast_periods + 1) * self.x[1]])).flatten()

        # assign to class
        self.cycles = NeuralDecomposition.inverse_scale_data(
            cycles, self.max_min_list, self.units)

        return None

    def get_trend(self):
        """Purpose: get trend component of time series
        It is structued in the same way as get_cycles()
        It is setting weights of cyclical component to zero
        """

        # take the trained model
        keras_model_trend = self.keras_model

        set_value(keras_model_trend.weights[0],
            (0 * np.floor(np.arange(self.n_data)))[np.newaxis, :].astype('float32')) # sin/kernel
        set_value(keras_model_trend.weights[1],
            (np.arange(self.n_data) * 0).astype('float32')) # sin/bias

        # compute trend component
        trend = keras_model_trend.predict(
            np.concatenate([self.x, 1 + np.arange(
                1, self.forecast_periods + 1) * self.x[1]])).flatten()

        # assign to class
        self.trend = NeuralDecomposition.inverse_scale_data(
            trend, self.max_min_list, self.units)

        return None

    @staticmethod
    def scale_data(data, units):
        """Purpose: scale data based on
        the number of units
        data - array of data, one dimensional
        units - number of units in the layer
        """

        min_value = data.min()
        max_value = data.max()

        # compute scaled output
        scaled_output = (data - min_value) / (max_value - min_value) * units

        return scaled_output, [max_value, min_value]

    @staticmethod
    def inverse_scale_data(data, max_min_list, units):
        """Purpose: inverse the scaled data obtained
        from the scale_data() function
        data - array of data, one dimmensional
        max_min_list - list of max and min values,
        obtained using scale_data() function
        max_min_list = [max_value, min_value]
        """

        # sanity check
        if len(max_min_list) != 2:
            raise TypeError('Input must be list and len == 2!')
        if max_min_list[1] > max_min_list[0]:
            raise ValueError('Check values in the input list!')

        # get min and max values
        max_value = max_min_list[0]
        min_value = max_min_list[1]

        inv_scaled_data = data * (max_value - min_value) / units + min_value

        return inv_scaled_data