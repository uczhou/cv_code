import numpy as np

MAX_ITERATIONS = 500


def advanced_initialize_centroids_value(k):
    # To be implemented
    return []


def initialize_centroids_value(k):
    # To be implemented
    return []


def get_centroids_number():
    # To be implemented
    return 5


def get_labels(data_set, centroids):

    distances = np.zeros((data_set.shape[0], centroids.shape[0]))

    for i in range(centroids.shape[0]):
        distances[:, i] = np.sqrt(((data_set - centroids[i])**2).sum(axis=0))

    # labels = np.zeros((data_set.shape[0], 1))

    labels = np.argmin(distances, axis=1)
    return labels


def get_centroids(data_set, labels, k):

    centroids = np.zeros((k, data_set.shape[0]))

    for i in range(k):
        centroids[i] = np.mean(data_set[labels == i], axis=0)
    return centroids


def error(old_centroids, centroids):
    return np.linalg.norm(centroids - old_centroids)


def is_done(old_centroids, centroids, iterations):
    if iterations > MAX_ITERATIONS:
        return True

    val = error(old_centroids, centroids)

    return val == 0


def kmeans(data_set, k):

    dimensions = data_set.shape

    centroids = initialize_centroids_value(dimensions, k)

    iterations = 0
    old_centroids = None

    while not is_done(old_centroids, centroids, iterations):

        old_centroids = centroids

        iterations += 1

        labels = get_labels(data_set, centroids)

        centroids = get_centroids(data_set, labels, k)

    return centroids




