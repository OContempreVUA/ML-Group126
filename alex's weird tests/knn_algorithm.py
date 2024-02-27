import numpy as np
import sympy as sym
import pandas as pd
from IPython.display import display, Math
from typing import Iterable, List, Dict, Tuple, Union, Optional
import math
sym.init_printing()

def euclidian_distance2(feature_set_data: Iterable[Union[int, float]], feature_set_prediction: Iterable[Union[int, float]]) -> float:
    """
    param feature_set_data: should contain an iterable datatype with the features a point in the training data has.
    param feature_set_prediction: should contain an iterable datatype with the features point from which we want to predict the class.
    It doesn't really mattter if you swap the two tho.
    """

    # set up some basic thingies
    distance: float = 0

    # put the different values for the same feature together in a tuple. Then store those tuples in a list
    zipped = list(zip(feature_set_data, feature_set_prediction))
    # here we add the squared difference between the features
    for feature in zipped:
        distance+= (float(feature[0])-feature[1])**2
    
    #return the square root of this difference
    return math.sqrt(distance)


class KNN_classifier:
    def __init__(self, k: int = 5):
        """Sets up the KNN classifier. 
        It requires the user to select a value for k, or go with the default value of 5."""
        self.k = 5
        self.data = None

    def feed_data(self, data: np.ndarray) -> None:
        """Takes in a iterable which contains nested interables.
        Each nested iterable stands for a row/instance in the dataset.
        And each (numerical) value stands for the value of a feature for that given instance.
        This method requires you to insert the data as a numpy array.

        !! THE LAST COLUMN MUST BE THE LABEL SET !!
        """

        self.data = data
        return None

    def add_row(self, row: np.ndarray) -> None:
        """allows user to add another row of data"""
        if not self.data:
            self.data=np.array(row)
        else:
            self.data.vstack(row)

        return None
    
    def predict(self, pred_point: Iterable[Union[int, float]]) -> str:
        """Here the class uses the KNN algorithm to predict whether class a particular point belongs to"""

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
            
        # now we count the amount of neighbors each class has
        class_proportion_dict: Dict[Iterable[Union[float, int, str]], float] = {}
        for neighbor in closest_points:
            if neighbor[-1] not in class_proportion_dict:
                class_proportion_dict[neighbor[-1]] = 1.0
            else:
                class_proportion_dict[neighbor[-1]] += 1

        # and we end with returning the class which has the highest tally
        highest_tally: int = 0
        highest_class: str = None

        for c in class_proportion_dict:
            if class_proportion_dict[c] > highest_tally:
                highest_tally = class_proportion_dict[c]
                highest_class = c
            
            else:
                continue
        
        return highest_class