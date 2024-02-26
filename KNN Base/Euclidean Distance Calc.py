import numpy as np

#knn(train_data, labels_train, test_data, k).

def euclidean_distance(p1, p2):
    """
    Calculate the Euclidean distance between two points.
    p1: first point in array or list.
    p2: second point in array or list.
    Returns: Euclidean Distance between p1 and p2.
    """
    point1 = np.array(p1)
    point2 = np.array(p2)
    return np.sqrt(np.sum(point1 - point2) ** 2)

def distances_from_point_to_set(point, set_of_points):
    distances = []
    for set_point in set_of_points:
        distance = euclidean_distance(point, set_point)
        distances.append(distance)
    return np.array(distances)


#testing
np.random.seed(126)
set_of_points = np.random.rand(100, 2) * 100
point = (50,50)

distances = distances_from_point_to_set(point, set_of_points)
print("point is at:")
print(point)
print("Distances from the point to each point in the set:")
print(distances)
