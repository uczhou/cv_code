import numpy as np
from matplotlib import pyplot as plt
import time

MAX_ITERATIONS = 500


def advanced_initialize_centroids_value(k):
    # To be implemented
    return []


def initialize_centroids_value(data_set, k):
    # To be implemented
    mean = np.mean(data_set, axis=0)
    std = np.std(data_set, axis=0)
    centroids = np.random.randn(k, data_set.shape[1]) * std + mean
    return centroids


def get_centroids_number():
    # To be implemented
    return 3


def get_labels(data_set, centroids):

    distances = np.zeros((data_set.shape[0], centroids.shape[0]))

    for i in range(centroids.shape[0]):
        distances[:, i] = np.sqrt(((data_set - centroids[i])**2).sum(axis=1))
    labels = np.argmin(distances, axis=1)
    return labels


def get_centroids(data_set, labels, k):

    centroids = np.zeros((k, data_set.shape[1]))

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

    centroids = initialize_centroids_value(data_set, k)

    iterations = 0
    old_centroids = np.zeros(centroids.shape)

    while not is_done(old_centroids, centroids, iterations):

        old_centroids = centroids

        iterations += 1

        labels = get_labels(data_set, centroids)

        centroids = get_centroids(data_set, labels, k)

        plt.clf()

        plt.scatter(data_set[:, 0], data_set[:, 1], s=7)

        plt.scatter(centroids[:, 0], centroids[:, 1], marker='.', c='y', s=150)
        plt.scatter(old_centroids[:, 0], old_centroids[:, 1], marker='.', c='r', s=150)

        plt.draw()
        plt.pause(1)

    return centroids


def gen_samples():
    center_1 = np.array([1, 1])
    center_2 = np.array([5, 5])
    center_3 = np.array([8, 1])

    # Generate random data and center it to the three centers
    data_1 = np.random.randn(200, 2) + center_1
    data_2 = np.random.randn(200, 2) + center_2
    data_3 = np.random.randn(200, 2) + center_3

    data = np.concatenate((data_1, data_2, data_3), axis=0)

    centroids = np.array([center_1, center_2, center_3])

    return centroids, data


def main():

    centroids, data_set = gen_samples()

    k = get_centroids_number()

    trained_centroids = kmeans(data_set, k)

    plt.clf()

    plt.scatter(data_set[:, 0], data_set[:, 1], s=7)

    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='g', s=150)
    plt.scatter(trained_centroids[:, 0], trained_centroids[:, 1], marker='*', c='r', s=150)

    plt.draw()
    plt.pause(5)


if __name__ == '__main__':
    main()
