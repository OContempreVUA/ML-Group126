import numpy as np
import sympy as sym
import pandas as pd
from IPython.display import display, Math
from typing import Iterable, List, Dict, Tuple, Union, Optional
import math
import random as rand
import matplotlib.pyplot as mpp
from knn_algorithm import KNN_classifier, K_Error, DimensionalityError, euclidian_distance2
import random

class Custom_KNN_Classifier(KNN_classifier):
    def __init__(self, k: int):
        if k < 4:
            raise K_Error()
        super().__init__()
        
        self.k = k
        
    def __str__(self) -> str:
        return f"KNN classifier with k={self.k}"

    def predict(self, pred_point: Iterable[int | float], verbose: bool = False) -> str:
        """Here the class uses the KNN algorithm to predict whether class a particular point belongs to"""

        if self.data.shape[1]-1 != len(pred_point):
            raise DimensionalityError()

        distance_neigbor_dict: Dict[float, Iterable[Union[float, int, str]]] = {}

        # here we go over each instance in the dataset 
        for instance in self.data:

            # and calculate the distance
            distance = euclidian_distance2(feature_set_data=instance, feature_set_prediction=pred_point)

            distance_neigbor_dict[distance] = instance


        ############---------WORKS---------############
        sorted_distances = sorted(list(distance_neigbor_dict.keys()))
        neighbor_distance_subset: dict[Iterable[Union[int, float, str]]: float] = dict()

        index: int = 0
        while len(neighbor_distance_subset) != self.k:
            next_closest_distance = sorted_distances[index]
            neighbor_distance_subset[tuple(distance_neigbor_dict[next_closest_distance])] = next_closest_distance
            index+=1
            
        # get furthest neighbor
        furthest_distance = max(neighbor_distance_subset.values())
        class_percentage_dict: dict[str: float] = {}
        class_proportion_dict: dict[str: float] = {}
        for neighbor in neighbor_distance_subset:
            
            # calculate for each neighbor the weight and sum up the weights per class
            if neighbor[-1] not in class_percentage_dict:
                
                class_percentage_dict[neighbor[-1]] = (furthest_distance - neighbor_distance_subset[neighbor])/furthest_distance
            else:
                class_percentage_dict[neighbor[-1]]  += (furthest_distance - neighbor_distance_subset[neighbor])/furthest_distance
                
            # also count the amount of neighbors per class
            if neighbor[-1] not in class_proportion_dict:
                class_proportion_dict[neighbor[-1]] = 1
            else:
                class_proportion_dict[neighbor[-1]] += 1
        #    print(f"this neighbor's weight = {(furthest_distance - neighbor_distance_subset[neighbor])/furthest_distance} from class: {neighbor[-1]}")
        #print(class_percentage_dict)
        #print(class_proportion_dict)
        if verbose:
            print(f"-"*20)
            for label in class_proportion_dict:
                print(f"{label} has a class proportion of: {round(number=class_proportion_dict[label], ndigits=2)} and a total weight of: {class_percentage_dict[label]}")
            print(f"-"*20)
            
        for label in class_proportion_dict:
            class_proportion_dict[label] /= self.k
        
        # calculate the class with the highest value:
        highest_class: str = None
        highest_score: float = 0
        
        for label in class_percentage_dict:
            score = class_percentage_dict[label] * class_proportion_dict[label]
            if verbose:
                print(f"{label} has a strength of: {score}")
            if score > highest_score:
                highest_score = score
                highest_class = label
        if verbose:
            print("-"*20)
        return highest_class
        
"""            
dataset = np.random.randint(low = -5, high = 100, size= [50,2])
colors: list = []
for _ in range(1,51):
    random_num = random.random()
    
    if random_num < 0.5:
        colors.append("green")
    else:
        colors.append("red")     
        
dataset = np.column_stack((dataset, colors))  
        
# print(dataset)

knn_custom = Custom_KNN_Classifier(k=7)
print(knn_custom.k)
knn_custom.feed_data(data=dataset)
knn_custom.predict(pred_point=np.array([5,7]))

dataframe = pd.DataFrame(data=dataset)

dataframe.to_csv("some_dataset.csv")
"""







