"""
Example contains model ML k-neighbors, using sklearn packet
"""

import numpy as np
import pandas as pd

from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


RESOURCES_PATH = '../resources/classified_data.csv'
LABEL_COL = 'TARGET CLASS'


def pre_process(raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:

    scaler = StandardScaler()

    scaler.fit(raw_data.drop(LABEL_COL, axis=1))

    scaled_features = scaler.transform(raw_data.drop(LABEL_COL, axis=1))

    scaled_data = pd.DataFrame(scaled_features, columns=raw_data.drop(LABEL_COL, axis=1).columns)

    return scaled_data, raw_data[LABEL_COL]


def test_k_neighbors() -> None:

    raw_data = pd.read_csv(RESOURCES_PATH, index_col=0)

    x, y = pre_process(raw_data=raw_data)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    error_rates = []

    for i in range(1, 100):
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        error_rates.append(np.mean(prediction != y_test))

    best_neighbors = error_rates.index(min(error_rates))

    model = KNeighborsClassifier(n_neighbors=best_neighbors)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)

    accuracy = np.mean(prediction == y_test)
    error = np.mean(prediction != y_test)

    print(f'Best accuracy for {best_neighbors}: {accuracy}')
    print(f'Least error for {best_neighbors}: {error}')

    assert accuracy > 0.9

