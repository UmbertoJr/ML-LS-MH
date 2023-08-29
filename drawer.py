import matplotlib.pyplot as plt
import numpy as np


def plot_points(pos):
    plt.scatter(pos[:, 0], pos[:, 1])
    plt.show()


def plot_points_sol(pos, X):
    plt.scatter(pos[:, 0], pos[:, 1])
    for node, [a, b] in enumerate(X):
        if a != -1:
            plt.plot([pos[node, 0], pos[a, 0]], [pos[node, 1], pos[a, 1]], 'k-')
        if b != -1:
            plt.plot([pos[node, 0], pos[b, 0]], [pos[node, 1], pos[b, 1]], 'k-')
    plt.show()


def plot_points_sol_intermediate(pos, X, X_int, tour):
    """
    pos: array of coordinates of the nodes
    X: TSP solution (list of pairs of nodes for each node)
    X_int: list of pairs of nodes for each node, with the intermediate nodes chosen by ML first phase
    tour: list of nodes in the tour
    """
    plt.scatter(pos[:, 0], pos[:, 1], s=80, alpha=0.8)
    n = len(tour) - 1
    
    # for node, [a, b] in enumerate(X):
    #     plt.plot([pos[node, 0], pos[a, 0]], [pos[node, 1], pos[a, 1]], 'k-')
        # plt.plot([pos[node, 0], pos[b, 0]], [pos[node, 1], pos[b, 1]], 'k-')
    for a, b in zip(tour[:-1], tour[1:]):
        plt.plot([pos[b, 0], pos[a, 0]], [pos[b, 1], pos[a, 1]], 'k-')
        plt.annotate(a, [pos[a,0], pos[a,1]])


    for node, [a, b] in enumerate(X_int):
        if a != -1:
            ind_a = np.argwhere(tour==a)[0][0]
            neigs = [tour[:-1][ind_a + 1 -n], tour[:-1][ind_a -1]]
            if a in neigs:
                """if a is a neighbor of node in the tour"""
                plt.plot([pos[node, 0], pos[a, 0]], [pos[node, 1], pos[a, 1]], 'b-')
            else:
                """if a is not a neighbor of node in the tour"""
                plt.plot([pos[node, 0], pos[a, 0]], [pos[node, 1], pos[a, 1]], 'r-')

        if b != -1:
            ind_b = np.argwhere(tour==b)[0][0]
            neigs = [tour[:-1][ind_b + 1 -n], tour[:-1][ind_b -1]]
            if b in neigs:
                plt.plot([pos[node, 0], pos[b, 0]], [pos[node, 1], pos[b, 1]], 'b-')
            else:
                plt.plot([pos[node, 0], pos[b, 0]], [pos[node, 1], pos[b, 1]], 'r-')
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    # plt.savefig('filename.eps', format='eps')
    # input()
    plt.show()
