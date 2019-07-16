from abc import ABCMeta, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
import numpy as np


class Algorithm(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, x_train, y_train):
        pass

    @abstractmethod
    def predict_proba(self, x_test):
        pass


class NeuralNetworkAlgorithm(Algorithm):

    batch_size = 256

    def __init__(self):
        self.model = None

    def fit(self, x_train, y_train):

        epochs = 250

        num_classes = y_train.shape[1]
        input_dimension = int(x_train.shape[1])

        self.model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(input_dimension,), dtype='float32'),
            keras.layers.Dropout(rate=0.2),
            keras.layers.Dense(2000, activation=keras.activations.relu),
            keras.layers.Dropout(rate=0.5),
            keras.layers.Dense(1000, activation=keras.activations.relu),
            keras.layers.Dropout(rate=0.5),
            keras.layers.Dense(1000, activation=keras.activations.relu),
            keras.layers.Dropout(rate=0.5),
            keras.layers.Dense(1000, activation=keras.activations.relu),
            keras.layers.Dropout(rate=0.5),
            keras.layers.Dense(1000, activation=keras.activations.relu),
            keras.layers.Dropout(rate=0.5),
            keras.layers.Dense(1000, activation=keras.activations.relu),
            keras.layers.Dropout(rate=0.5),
            keras.layers.Dense(500, activation=keras.activations.relu),
            keras.layers.Dropout(rate=0.5),
            keras.layers.Dense(num_classes, activation=keras.activations.softmax)
        ])

        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adam(lr=0.0005),
                           metrics=['accuracy'])

        # Add Early Stopping
        callbacks = [keras.callbacks.EarlyStopping(monitor='acc', patience=15)]

        return self.model.fit(x_train, y_train,
                              batch_size=self.batch_size,
                              epochs=epochs,
                              callbacks=callbacks,
                              verbose=0)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test, batch_size=self.batch_size)


class RandomForestClassifierAlgorithm(Algorithm):

    def __init__(self):
        self.model = None

    def fit(self, x_train, y_train):
        self.model = RandomForestClassifier(n_estimators=100)
        return self.model.fit(x_train, y_train)

    def predict_proba(self, x_test):
        y_pred_proba = self.model.predict_proba(x_test)
        return np.transpose(y_pred_proba)[1]


def algorithm_switch(name: str) -> Algorithm:
    switcher = {
        'NeuralNetworkAlgorithm': NeuralNetworkAlgorithm,
        'RandomForestClassifierAlgorithm': RandomForestClassifierAlgorithm
    }

    algorithm = switcher.get(name, lambda: "Selector not found!")
    return algorithm()

