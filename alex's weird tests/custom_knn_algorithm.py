import numpy as np
import sympy as sym
import pandas as pd
from IPython.display import display, Math
from typing import Iterable, List, Dict, Tuple, Union, Optional
import math
import random as rand
import matplotlib.pyplot as mpp
from knn_algorithm import KNN_classifier, K_Error, DimensionalityError, euclidian_distance2


class Custom_KNN_Classifier(KNN_classifier):
    def __init__(self, k: int):
        if k < 4:
            raise K_Error()
        super().__init__(k)

    def predict(self, pred_point: Iterable[int | float]) -> str:
        """Here the class uses the KNN algorithm to predict whether class a particular point belongs to"""

        if self.data.shape[1]-1 != len(pred_point):
            raise DimensionalityError()

        distances: Dict[float, Iterable[Union[float, int, str]]] = {}

        # here we go over each instance in the dataset 
        for instance in self.data:

            # and calculate the distance
            distance = euclidian_distance2(feature_set_data=instance, feature_set_prediction=pred_point)

            distances[distance] = instance


        ############---------WORKS---------############
        sorted_distances = sorted(list(distances.keys()))
        closest_points: List[Iterable[Union[float, int, str]]] = []

        index: int = 0
        while len(closest_points) != self.k:
            smallest_distance = sorted_distances[index]
            closest_points.append(distances[smallest_distance])
            index+=1

        print(closest_points)

        # calculate class proportion
        class_proportion_dict: Dict[str, int]
        for neighbor in closest_points:
            if neighbor[-1] not in class_proportion_dict:
                class_proportion_dict[neighbor[-1]] = 1
            else:
                class_proportion_dict[neighbor[-1]] += 1
        
        # get furthest neighbor

    







dataset: np.ndarray = np.random.randint(low=0, high = 10, size=[20,3])
labels: List[str] = []

for _ in range(0,dataset.shape[0]):
    color: str = None
    random_num: int = rand.random()
    if random_num <0.5:
        color = "blue"
    elif random_num >0.8:
        color = "red"
    else:
        color = "green"
    labels.append(color)



full_dataset = np.column_stack((dataset, labels))
knn = Custom_KNN_Classifier(k=5)
knn.feed_data(full_dataset)
print(knn.predict([3,3,3]))
