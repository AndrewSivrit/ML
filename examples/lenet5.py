import keras.layers as layers
import numpy as np
import tensorflow as tf

from tensorflow import keras
from typing import Tuple


NUM_CLASSES = 10


class LeNet(keras.Sequential):   # type: ignore
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 num_classes: int = 10
                 ) -> None:
        super().__init__()
        self.add(
            layers.Conv2D(
                filters=6,
                kernel_size=(3, 3),
                activation='relu',
                input_shape=input_shape
            )
        )
        self.add(layers.AveragePooling2D())
        self.add(layers.Conv2D(filters=16,
                               kernel_size=(3, 3),
                               activation='relu')
                 )
        self.add(layers.AveragePooling2D())
        self.add(layers.Flatten())
        self.add(layers.Dense(units=120, activation='relu'))
        self.add(layers.Dense(units=84, activation='relu'))
        self.add(layers.Dense(units=num_classes, activation='softmax'))


def preproc_data_x(data: np.ndarray) -> np.ndarray:     # type: ignore

    data = data.reshape([data.shape[0], data.shape[1], data.shape[2], 1])

    data = np.pad(data, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

    data = data.astype('float32')

    data /= 255

    return data


def test_lenet5() -> None:

    model = LeNet(input_shape=(32, 32, 1), num_classes=10)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(
        path="mnist.npz")

    x_train = preproc_data_x(x_train)
    x_test = preproc_data_x(x_test)

    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    model.fit(x_train, y_train, steps_per_epoch=10, epochs=10)

    score = model.evaluate(x_test, y_test, batch_size=1)[-1]

    assert score > 0.9
