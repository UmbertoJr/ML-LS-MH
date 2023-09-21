"""
This file contains the function plot_points_sol_intermediate which is used to plot the TSP solution with intermediate nodes chosen by ML first phase.
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_points(pos):
    """
    Plots the given points on a 2D scatter plot.

    Args:
        pos (numpy.ndarray): A 2D numpy array containing the x and y coordinates of the points to be plotted.

    Returns:
        None
    """
    plt.scatter(pos[:, 0], pos[:, 1])

    plt.show()


def plot_points_sol(pos, X):
    # scatter plot of the nodes
    plt.scatter(pos[:, 0], pos[:, 1])
    # plot the edges of the TSP solution
    for node, [a, b] in enumerate(X):
        if a != -1:
            plt.plot([pos[node, 0], pos[a, 0]], [pos[node, 1], pos[a, 1]], 'k-')
        if b != -1:
            plt.plot([pos[node, 0], pos[b, 0]], [pos[node, 1], pos[b, 1]], 'k-')
    # show the plot
    plt.show()


def plot_points_sol_intermediate(pos, X, X_int, tour):
    """
    Plot the TSP solution with intermediate nodes chosen by ML first phase.
    
    Parameters:
    pos (numpy.ndarray): array of coordinates of the nodes
    X (list): TSP solution (list of pairs of nodes for each node)
    X_int (list): list of pairs of nodes for each node, with the intermediate nodes chosen by ML first phase
    tour (list): list of nodes in the tour
    """
    # plot nodes
    plt.scatter(pos[:, 0], pos[:, 1], s=80, alpha=0.8)

    # plot edges of the tour
    for a, b in zip(tour[:-1], tour[1:]):
        plt.plot([pos[b, 0], pos[a, 0]], [pos[b, 1], pos[a, 1]], 'k-')
        plt.annotate(a, [pos[a,0], pos[a,1]])

    # plot edges of the intermediate nodes
    for node, [a, b] in enumerate(X_int):
        if a != -1:
            # get the index of node a in the tour
            ind_a = np.argwhere(tour==a)[0][0]
            # get the neighbors of node a in the tour
            neigs = [tour[:-1][ind_a + 1 - len(tour)], tour[:-1][ind_a - 1]]
            if a in neigs:
                # if a is a neighbor of node in the tour
                plt.plot([pos[node, 0], pos[a, 0]], [pos[node, 1], pos[a, 1]], 'b-')
            else:
                # if a is not a neighbor of node in the tour
                plt.plot([pos[node, 0], pos[a, 0]], [pos[node, 1], pos[a, 1]], 'r-')

        if b != -1:
            # get the index of node b in the tour
            ind_b = np.argwhere(tour==b)[0][0]
            # get the neighbors of node b in the tour
            neigs = [tour[:-1][ind_b + 1 - len(tour)], tour[:-1][ind_b - 1]]
            if b in neigs:
                # if b is a neighbor of node in the tour
                plt.plot([pos[node, 0], pos[b, 0]], [pos[node, 1], pos[b, 1]], 'b-')
            else:
                # if b is not a neighbor of node in the tour
                plt.plot([pos[node, 0], pos[b, 0]], [pos[node, 1], pos[b, 1]], 'r-')

    # remove x and y axis
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)

    # show the plot
    plt.show()


