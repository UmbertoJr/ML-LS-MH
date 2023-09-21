import calendar
import time
import numpy as np
from utils import create_tour_from_X
from candidate_list import CandidateList
from machine_learning_models import MLAdd
from utils import compute_tour_lenght
import multiprocessing as mp

class MLGreedy:

    @staticmethod
    def inner_loop(X, edge):
        [i, j] = edge
        n_nodes = 1
        n = X.shape[0]
        X[i] = np.array([j, X[i, 0]])
        X[j] = np.array([i, X[j, 0]])
        a, b = X[i, 0], i
        while n_nodes < n:
            # print(f'{n_nodes} <> {b}->{a}')
            if a == i:
                return True
            if a == -1:
                break
            if np.sum(X[a] == -1) == 2:
                break
            tmp = a
            a = X[a, 0] if X[a, 0] != b else X[a, 1]
            b = tmp
            n_nodes = n_nodes + 1
        return False

    @staticmethod
    def run(n, positions, distance_matrix, optimal_tour, 
            cl_method=CandidateList.Method.NearestNeighbour,
            ml_model=MLAdd.MLModel.NearestNeighbour, limit=15, opt_len=None, 
            improvement_type="2-Opt", style="complete", name_instance=""):
        
        t0 = time.time()
        # create CL for each vertex
        candidate_list = CandidateList.compute(positions, distance_matrix, cl_method)
        
        
        # insert the shortest two vertices for each CL into L_P
        L_P = np.empty((2, n * 2), dtype=int)
        L_P_distances = np.empty(n * 2, dtype=int)
        for node in range(n):
            #vertices = candidate_list[node][np.argsort(distance_matrix[node, candidate_list[node]])[:2]]
            vertices = np.argsort(distance_matrix[node])[1:3]
            L_P[:, node] = [node, vertices[0]]
            L_P[:, n + node] = [node, vertices[1]]


        # sort L_P according to ascending costs c_{i,j} (before the first, then the second)
        costs_1st = np.array([distance_matrix[i, j] for [i, j] in L_P.T[:n]])
        costs_2nd = np.array([distance_matrix[i, j] for [i, j] in L_P.T[n:]])
        L_P = L_P[:, np.concatenate((np.argsort(costs_1st), np.argsort(costs_2nd) + n))]
        L_P_distances[:n] = [1] * n
        L_P_distances[n:] = [2] * n
        
        
        # initialize X
        X = np.full((n, 2), -1, dtype=int)


        # initialize ML Model
        ML_add = MLAdd(model=ml_model)


        # for the edge in L_P select the extreme vertices i, j of the edge
        for L_P_pos, [i, j] in enumerate(L_P.T):
            
            # if i and j have less than two connections each in X
            if np.sum(X[i] > -1) < 2 and np.sum(X[j] > -1) < 2:


                # if l do not creates a inner-loop
                if not MLGreedy.inner_loop(X.copy(), [i, j]):
                    
                    # creates the local view context used by the ML model
                    nodes = np.concatenate(([i], candidate_list[i][:limit]))
                    nodes = np.pad(nodes, (0, limit + 1 - len(nodes)), constant_values=(-1))
                    dists = np.full(int(limit * (limit + 1) / 2), fill_value=-1, dtype=np.float64)
                    edges = np.empty(int(limit * (limit + 1) / 2), dtype=object)
                    edges_in_sol = np.zeros(int(limit * (limit + 1) / 2))
                    current_pos = 0
                    for pos_node_a, node_a in enumerate(nodes[:-1]):
                        for pos_node_b, node_b in enumerate(nodes[pos_node_a + 1:]):
                            if node_a == -1 or node_b == -1:
                                edges[current_pos] = (-1, -1)
                                continue
                            dists[current_pos] = distance_matrix[node_a][node_b]
                            edges[current_pos] = (node_a, node_b)
                            edges_in_sol[current_pos] = node_b in X[node_a]
                            current_pos += 1

                    pos_i_opt = np.argwhere(optimal_tour == i)[0][0]
                    ret = (j == optimal_tour[pos_i_opt - 1] or j == optimal_tour[(pos_i_opt + 1) % len(optimal_tour)])

                    # if the ML agrees the addition of l
                    if ML_add(distance=L_P_distances[L_P_pos], distance_vector=dists, 
                              solution_vector=edges_in_sol, in_opt=ret, 
                              i=i, j=j, name=name_instance):
                        X[i] = np.array([j, X[i, 0]])
                        X[j] = np.array([i, X[j, 0]])

        X_intermediate = np.copy(X)

        # find the hub vertex h: h = argmax_{i \in V} TD[i], where TD is the total
        # distance e.g. the sum of all the distances outgoing from a node
        TD = np.sum(distance_matrix, axis=0)
        h = np.argmin(TD)


        # select all the edges that connects free vertices and insert them into L_D
        free_vertices = np.where(np.sum(X == -1, axis=1))[0]
        free_vertices_masked = np.ma.array(free_vertices, mask=False)
        L_D = np.array([[], []], dtype=int)
        for i, vert in enumerate(free_vertices[:-1]):
            free_vertices_masked.mask[i] = True
            L_D = np.hstack((L_D, [np.full(np.ma.count(free_vertices_masked), vert),
                                   free_vertices_masked.compressed()]))
        L_D = L_D.T
        # compute the savings values wrt h for each edge in L_D,
        # where s_{i, j} = c_{i, h} + c_{h, j} - c_{i, j}
        s = np.array([distance_matrix[i, h] + distance_matrix[h, j] - distance_matrix[i, j] for [i, j] in L_D])
        # sort L_D according to the descending savings s_{i, j}
        L_D = L_D[np.argsort(-s)]
        t = 0
        # while the solution X is not complete
        while (X == -1).any():
            # select the extreme vertices i, j of l
            [i, j] = L_D[t]
            t = t + 1
            # if vertex i and vertex j have less than two connections each in X
            if np.sum(X[i] > -1) < 2 and np.sum(X[j] > -1) < 2:
                # if l do not creates a inner-loop
                if not MLGreedy.inner_loop(X.copy(), [i, j]):
                    X[i] = np.array([j, X[i, 0]])
                    X[j] = np.array([i, X[j, 0]])
        time_mlg = time.time() - t0


        if improvement_type == "2-Opt":
            results_to_return = MLGreedy.improve_solution(X, 
                                                          X_intermediate,
                                                          distance_matrix, 
                                                          candidate_list,
                                                          style=style,
                                                          opt_len=opt_len)
        elif improvement_type == "2-Opt-CL":
            results_to_return = MLGreedy.improve_solution(X, 
                                                          X_intermediate, 
                                                          distance_matrix, 
                                                          candidate_list, 
                                                          with_CL=True,
                                                          style=style,
                                                          opt_len=opt_len)
        elif improvement_type == "ILS":
            results_to_return = MLGreedy.improve_ILS_solution(X, 
                                                              X_intermediate, 
                                                              distance_matrix, 
                                                              candidate_list, 
                                                              opt_len, 
                                                              style=style)

        # ml_tour = create_tour_from_X(X)
        # results_to_return = {
        #     "tour Constructive": ml_tour,
        #     "X Constructive": X, 

        #     f"tour NO simple": ml_tour,
        #     f"X NO simple": X,
            
        #     f"Time NO simple": 0,
        #     f"Ops NO simple": 0,
            
        #     "free_nodes": MLGreedy.get_free_nodes(X_intermediate),
        #     "fixed edges": MLGreedy.get_fixed_edges(X_intermediate),
        # }
    

        
        results_to_return["Time ML-C"] = time_mlg
        results_to_return["X Intermediate"] = X_intermediate
        return results_to_return
        # X_ILS, time_ILS, operation_ILS, calls_LS, 
        # return X, X_intermediate, X_improved, time_mlg, time_2opt
        # return X, X_intermediate, X, time_mlg, 0
        # return X, X_intermediate, X_improved, time_mlg, time_2opt
        # return X, X_intermediate, X_improved, time_mlg, time_2opt

    @staticmethod
    def get_free_nodes(X):
        """
        This function takes X intermediate and returns the nodes that has at least one connection free
        """
        """
        This function takes X intermediate and returns the nodes that has at least one connection free
        """
        free_nodes = np.array([i for i in range(len(X)) if (X[i] == -1).any()])
        return free_nodes

    @staticmethod
    def get_fixed_edges(X):
        """
        This function takes X_intermediate and returns in a list all the edges connected during first phase
        """
        """
        This function takes X_intermediate and returns in a list all the edges connected during first phase
        """
        fixed_edges = np.empty(np.sum(X != -1), dtype=object)
        n_fixed_e = 0
        for i, [a, b] in enumerate(X):
            if a != -1:
                fixed_edges[n_fixed_e] = (i, a)
                n_fixed_e += 1
            if b != -1:
                fixed_edges[n_fixed_e] = (i, b)
                n_fixed_e += 1
        return list(fixed_edges)

    @staticmethod
    def improve_solution(X, X_intermediate, distance_matrix, CLs,
                         with_CL = False, 
                         style="complete", opt_len=0):
        
        fixed_edges = MLGreedy.get_fixed_edges(X_intermediate)
        free_nodes = MLGreedy.get_free_nodes(X_intermediate)
        
        if style == "complete":
            if with_CL:
                improvement_function = MLGreedy.two_opt_complete_CL
            else:
                improvement_function = MLGreedy.two_opt_complete
        elif style == "free":
            if with_CL:
                improvement_function = MLGreedy.two_opt_free_CL
            else:
                improvement_function = MLGreedy.two_opt_free
        elif style == "reduced":
            if with_CL:
                improvement_function = MLGreedy.two_opt_reduced_CL
            else:
                improvement_function = MLGreedy.two_opt_reduced
                
        
        #  qui si lancia il 2-opt
        t0 = time.time()
        X_c = np.copy(X)
        tour_ = create_tour_from_X(X_c)
        len_tour = compute_tour_lenght(tour_, distance_matrix)
        count_ = 0
        ops_ = 0
        repeat = 0

        while True:
            X_c, improvement, ops_, tour_ = improvement_function(X_c,
                                                                 free_nodes, 
                                                                 fixed_edges,
                                                                 distance_matrix, 
                                                                 CLs,
                                                                 tour_)
            
            tour_ = tour_[::-1]
            count_ += ops_
            len_tour -= improvement
            # print(f'Process {mp.current_process().name} improvement = {improvement},'
            #       f'len_tour = {len_tour}, gap = {(len_tour - opt_len) / opt_len * 100:.3f} %')
            if improvement == 0:
                if (repeat == 1 or style=="complete"):    
                    time_ = time.time() - t0
                    break
                else:
                    repeat += 1    
                    repeat = repeat % 2
            else:
                repeat = 0


            # if with_CL:
            #     X_c_reduced, improvement, 
            #     ops_plus, tour_reduced = MLGreedy.two_opt_reduced_CL(X_c_reduced, 
            #                                                          free_nodes, 
            #                                                          fixed_edges,
            #                                                          distance_matrix, 
            #                                                          CLs, 
            #                                                          tour_reduced)
            # else:
            #     X_c_reduced, improvement, 
            #     ops_plus, tour_reduced = MLGreedy.two_opt_reduced(X_c_reduced, 
            #                                                       free_nodes, 
            #                                                       fixed_edges,
            #                                                       distance_matrix, 
            #                                                       CLs, 
            #                                                       tour_reduced)
                

                
                
                
        #     ops_reduced += ops_plus
        #     if improvement == 0:
        #         time_reduced = time.time() - t0
        #         break
        #     else:
        #         count_reduced += 1
        

        # #  qui si lancia il 2-opt free
        # t0 = time.time()
        # X_c_free = np.copy(X)
        # tour_free = create_tour_from_X(X_c_free)
        # count_free = 0
        # ops_free = 0
        # repeat_free = 0
        # while True:
        #     if with_CL:
        #         X_c_free, improvement, ops_plus, tour_free = MLGreedy.two_opt_free_CL(X_c_free, free_nodes, fixed_edges, 
        #                                                                               distance_matrix, CLs, tour_free)
        #     else:
        #         X_c_free, improvement, ops_plus, tour_free = MLGreedy.two_opt_free(X_c_free, free_nodes, fixed_edges, 
        #                                                                            distance_matrix, CLs, tour_free)
            
        #     tour_free = tour_free[::-1]
        #     ops_free += ops_plus
        #     repeat_free += 1
        #     repeat_free = repeat_free % 2
        #     if improvement == 0 and repeat_free == 1:
        #         time_free = time.time() - t0
        #         break
        #     else:
        #         count_free += 1

        
        # #  qui si lancia il 2-opt complete
        # t0 = time.time()
        # X_c_complete = np.copy(X) 
        # tour_complete = create_tour_from_X(X_c_complete)
        # count_complete = 0
        # ops_complete = 0
        
        # #  qui si lancia il 2-opt ridotto
        # t0 = time.time()
        # X_c_reduced = np.copy(X)
        # tour_reduced = create_tour_from_X(X_c_reduced)
        # count_reduced = 0
        # ops_reduced = 0

        # while True:
        #     if with_CL:
        #         X_c_reduced, improvement, ops_plus, tour_reduced = MLGreedy.two_opt_reduced_CL(X_c_reduced, free_nodes, fixed_edges,
        #                                                                                        distance_matrix, CLs, tour_reduced)
        #     else:
        #         X_c_reduced, improvement, ops_plus, tour_reduced = MLGreedy.two_opt_reduced(X_c_reduced, free_nodes, fixed_edges,
        #                                                                                     distance_matrix, CLs, tour_reduced)
                
        #     ops_reduced += ops_plus
        #     if improvement == 0:
        #         time_reduced = time.time() - t0
        #         break
        #     else:
        #         count_reduced += 1
        

        # #  qui si lancia il 2-opt free
        # t0 = time.time()
        # X_c_free = np.copy(X)
        # tour_free = create_tour_from_X(X_c_free)
        # count_free = 0
        # ops_free = 0
        # repeat_free = 0
        # while True:
        #     if with_CL:
        #         X_c_free, improvement, ops_plus, tour_free = MLGreedy.two_opt_free_CL(X_c_free, free_nodes, fixed_edges, 
        #                                                                               distance_matrix, CLs, tour_free)
        #     else:
        #         X_c_free, improvement, ops_plus, tour_free = MLGreedy.two_opt_free(X_c_free, free_nodes, fixed_edges, 
        #                                                                            distance_matrix, CLs, tour_free)
            
        #     tour_free = tour_free[::-1]
        #     ops_free += ops_plus
        #     repeat_free += 1
        #     repeat_free = repeat_free % 2
        #     if improvement == 0 and repeat_free == 1:
        #         time_free = time.time() - t0
        #         break
        #     else:
        #         count_free += 1

        
        data_to_return = {
            "tour Constructive": create_tour_from_X(X),
            "X Constructive": X, 

            f"tour 2-Opt-{['', 'CL'][with_CL]} {style}": tour_,
            f"X 2-Opt-{['', 'CL'][with_CL]} {style}": X_c,
            
            f"Time 2-Opt-{['', 'CL'][with_CL]} {style}": time_,
      
            f"Ops 2-Opt-{['', 'CL'][with_CL]} {style}":count_,
            
            "free_nodes": free_nodes,
            "fixed edges": fixed_edges,
        }
        return data_to_return
                
    @staticmethod
    def two_opt_reduced(X, free_nodes, fixed_edges, distance_matrix, CLs, tour):
        # tour = create_tour_from_X(X)
        # assert check_feasibility(tour), "Problem the tour is not feasible"
        ops = 0
        for i, node_ip in enumerate(tour[:-1]):
            node_in = tour[i + 1]
            if node_ip in free_nodes and node_in in free_nodes and (node_ip, node_in) not in fixed_edges:
                for j, node_jp in enumerate(tour[i+1:-2]):
                    node_jn = tour[i + j + 2]
                    if node_jp in free_nodes and node_jn in free_nodes and (node_jp, node_jn) not in fixed_edges:
                        old_cost = distance_matrix[node_ip][node_in] + distance_matrix[node_jp][node_jn]
                        new_cost = distance_matrix[node_ip][node_jp] + distance_matrix[node_in][node_jn]
                        ops += 1
                        # print(old_cost, new_cost)
                        # print(old_cost, new_cost)
                        if old_cost - new_cost > 0:
                            # print("\nentrato!")
                            # print(node_ip, node_in, node_jp, node_jn)
                            # print(f"improvement = {old_cost - new_cost}")
                            # print()
                            # print(X[node_ip], X[node_in], X[node_jp], X[node_jn])
                            # print(X)
                            # print("\nentrato!")
                            # print(node_ip, node_in, node_jp, node_jn)
                            # print(f"improvement = {old_cost - new_cost}")
                            # print()
                            # print(X[node_ip], X[node_in], X[node_jp], X[node_jn])
                            # print(X)
                            X[node_ip][np.where(X[node_ip] == node_in)[0][0]] = node_jp
                            X[node_in][np.where(X[node_in] == node_ip)[0][0]] = node_jn
                            X[node_jp][np.where(X[node_jp] == node_jn)[0][0]] = node_ip
                            X[node_jn][np.where(X[node_jn] == node_jp)[0][0]] = node_in
                            tour[i+1:i+j+2]  = np.flip(tour[i+1:i+j+2], axis=0)
                            # print(X[node_ip], X[node_in], X[node_jp], X[node_jn])
                            # # print(X)
                            # print(f"new tour {i, j}")
                            # print(tour)
                            return X, old_cost - new_cost, ops, tour
        return X, 0, ops, tour
    
    @staticmethod
    def two_opt_free(X, free_nodes, fixed_edges, distance_matrix, CLs, tour):
        # tour = create_tour_from_X(X)
        # assert check_feasibility(tour), "Problem the tour is not feasible"
        ops = 0
        for node_ip in free_nodes:
            i = np.argwhere(tour == node_ip)[0][0]
            node_in = tour[i+1]

            for j, node_jp in enumerate(tour[i+1:-2]):
                node_jn = tour[i + j + 2]
                old_cost = distance_matrix[node_ip][node_in] + distance_matrix[node_jp][node_jn]
                new_cost = distance_matrix[node_ip][node_jp] + distance_matrix[node_in][node_jn]
                ops += 1
                if old_cost - new_cost > 0:
                    # print(f"improvement = {old_cost - new_cost}")
                    X[node_ip][np.where(X[node_ip] == node_in)[0][0]] = node_jp
                    X[node_in][np.where(X[node_in] == node_ip)[0][0]] = node_jn
                    X[node_jp][np.where(X[node_jp] == node_jn)[0][0]] = node_ip
                    X[node_jn][np.where(X[node_jn] == node_jp)[0][0]] = node_in
                    tour[i+1:i+j+2]  = np.flip(tour[i+1:i+j+2], axis=0)
                    return X, old_cost - new_cost, ops, tour
        return X, 0, ops, tour

    @staticmethod
    def two_opt_complete(X, free_nodes, fixed_edges, distance_matrix, CLs, tour):
        # tour = create_tour_from_X(X)
        ops = 0
        for i, node_ip in enumerate(tour[:-1]):
            node_in = tour[i + 1]
            for j, node_jp in enumerate(tour[i+1:-2]):
                node_jn = tour[i + j + 2]
                old_cost = distance_matrix[node_ip][node_in] + distance_matrix[node_jp][node_jn]
                new_cost = distance_matrix[node_ip][node_jp] + distance_matrix[node_in][node_jn]
                ops += 1
                if old_cost - new_cost > 0:
                    # print(f'({node_ip}, {node_in}) <-> ({node_jp}, {node_jn})')
                    X[node_ip][np.where(X[node_ip] == node_in)[0][0]] = node_jp
                    X[node_in][np.where(X[node_in] == node_ip)[0][0]] = node_jn
                    X[node_jp][np.where(X[node_jp] == node_jn)[0][0]] = node_ip
                    X[node_jn][np.where(X[node_jn] == node_jp)[0][0]] = node_in
                    tour[i+1:i+j+2]  = np.flip(tour[i+1:i+j+2], axis=0)
                    # print(f"improvement = {old_cost - new_cost}")
                    return X, old_cost - new_cost, ops, tour
        return X, 0, ops, tour

    

    @staticmethod
    def two_opt_reduced_CL(X, free_nodes, fixed_edges, distance_matrix, CLs, tour):
        # tour = create_tour_from_X(X)
        # assert check_feasibility(tour), "Problem the tour is not feasible"
        n = len(tour)
        ops = 0
        for i, node_ip in enumerate(tour[:-1]):
            node_in = tour[i + 1]
            if node_ip in free_nodes and node_in in free_nodes and (node_ip, node_in) not in fixed_edges:
                for node_jp in CLs[node_ip]:
                    j = np.argwhere(tour == node_jp)[0][0]
                    if i < j:
                        node_jn = tour[j + 1- n]
                        if node_jp in free_nodes and node_jn in free_nodes and (node_jp, node_jn) not in fixed_edges:
                            old_cost = distance_matrix[node_ip][node_in] + distance_matrix[node_jp][node_jn]
                            new_cost = distance_matrix[node_ip][node_jp] + distance_matrix[node_in][node_jn]
                            ops += 1
                            # print(old_cost, new_cost)
                            if old_cost - new_cost > 0:
                                # print("\nentrato!")
                                # print(node_ip, node_in, node_jp, node_jn)
                                # print(f"improvement = {old_cost - new_cost}")
                                # print()
                                # print(X[node_ip], X[node_in], X[node_jp], X[node_jn])
                                # print(X)
                                X[node_ip][np.where(X[node_ip] == node_in)[0][0]] = node_jp
                                X[node_in][np.where(X[node_in] == node_ip)[0][0]] = node_jn
                                X[node_jp][np.where(X[node_jp] == node_jn)[0][0]] = node_ip
                                X[node_jn][np.where(X[node_jn] == node_jp)[0][0]] = node_in
                                # tour[i+1:j+1]  = np.flip(tour[i+1:j+1], axis=0)
                                # print(X[node_ip], X[node_in], X[node_jp], X[node_jn])
                                # # print(X)
                                # print(f"new tour {i, j}")
                                # print(tour)
                                return X, old_cost - new_cost, ops, create_tour_from_X(X)
        return X, 0, ops, tour
    
    @staticmethod
    def two_opt_free_CL(X, free_nodes, fixed_edges, distance_matrix, CLs, tour):
        # tour = create_tour_from_X(X)
        # assert check_feasibility(tour), "Problem the tour is not feasible"
        n = len(tour)
        ops = 0
        for node_ip in free_nodes:
            i = np.argwhere(tour == node_ip)[0][0]
            node_in = tour[i+1]
            for node_jp in CLs[node_ip]:
                j = np.argwhere(tour == node_jp)[0][0]
                if i < j:
                    node_jn = tour[j + 1- n]
                    old_cost = distance_matrix[node_ip][node_in] + distance_matrix[node_jp][node_jn]
                    new_cost = distance_matrix[node_ip][node_jp] + distance_matrix[node_in][node_jn]
                    ops += 1
                    if old_cost - new_cost > 0:
                        # print(node_in, node_ip, node_jp, node_jn)
                        # print(X[node_ip], X[node_in], X[node_jp], X[node_jn])
                        # print(f"improvement = {old_cost - new_cost}")
                        X[node_ip][np.where(X[node_ip] == node_in)[0][0]] = node_jp
                        X[node_in][np.where(X[node_in] == node_ip)[0][0]] = node_jn
                        X[node_jp][np.where(X[node_jp] == node_jn)[0][0]] = node_ip
                        X[node_jn][np.where(X[node_jn] == node_jp)[0][0]] = node_in
                        tour[i+1:j+1]  = np.flip(tour[i+1:j+1], axis=0)
                        # print(i,j)
                        # print(X[node_ip], X[node_in], X[node_jp], X[node_jn])
                        # print(tour)
                        # print()
                        return X, old_cost - new_cost, ops, tour
        return X, 0, ops, tour

    @staticmethod
    def two_opt_complete_CL(X, free_nodes, fixed_edges, distance_matrix, CLs, tour):
        # tour = create_tour_from_X(X)
        n = len(tour)
        ops = 0
        for i, node_ip in enumerate(tour[:-1]):
            node_in = tour[i + 1]
            for node_jp in CLs[node_ip]:
                j = np.argwhere(tour == node_jp)[0][0]
                if i < j:
                    node_jn = tour[j + 1- n]
                    old_cost = distance_matrix[node_ip][node_in] + distance_matrix[node_jp][node_jn]
                    new_cost = distance_matrix[node_ip][node_jp] + distance_matrix[node_in][node_jn]
                    ops += 1
                    if old_cost - new_cost > 0:
                        # print(f'({node_ip}, {node_in}) <-> ({node_jp}, {node_jn})')
                        X[node_ip][np.where(X[node_ip] == node_in)[0][0]] = node_jp
                        X[node_in][np.where(X[node_in] == node_ip)[0][0]] = node_jn
                        X[node_jp][np.where(X[node_jp] == node_jn)[0][0]] = node_ip
                        X[node_jn][np.where(X[node_jn] == node_jp)[0][0]] = node_in
                        # tour[i+1:j+1]  = np.flip(tour[i+1:j+1], axis=0)
                        # print(f"improvement = {old_cost - new_cost}")
                        return X, old_cost - new_cost, ops, create_tour_from_X(X)
        return X, 0, ops, tour

    @staticmethod
    def two_cl(i, X_i, tour, free_nodes, fixed_edges, distance_matrix, CLs):
        n = len(tour)
        X = np.copy(X_i)
        tour = create_tour_from_X(X)
        node_ip = tour[i]
        node_in = tour[i + 1 - n]
        ops = 0
        if node_ip in free_nodes and node_in in free_nodes and (node_ip, node_in) not in fixed_edges:    
            for node_jp in np.random.permutation(CLs[node_ip]):    
                j = np.argwhere(tour[:-1] == node_jp)[0][0]
                node_jn = tour[j + 1]
                if node_jp in free_nodes and node_jn in free_nodes and (node_jp, node_jn) not in fixed_edges:
                    ops += 1
                    old_cost = distance_matrix[node_ip][node_in] + distance_matrix[node_jp][node_jn]
                    new_cost = distance_matrix[node_ip][node_jp] + distance_matrix[node_in][node_jn]
                    # print(f"Process {mp.current_process().name}")
                    # print(f"node_ip {node_ip}, node_in {node_in}")
                    # print(f"node_jp {node_jp}, node_jn {node_jn}")
                    # print(X[node_ip], X[node_in], X[node_jp], X[node_jn])
                    # print(tour)
                    X[node_ip][np.where(X[node_ip] == node_in)[0][0]] = node_jp
                    X[node_in][np.where(X[node_in] == node_ip)[0][0]] = node_jn
                    X[node_jp][np.where(X[node_jp] == node_jn)[0][0]] = node_ip
                    X[node_jn][np.where(X[node_jn] == node_jp)[0][0]] = node_in
                    tour[i+1:j+1]  = np.flip(tour[i+1:j+1], axis=0)
                    return X, old_cost - new_cost, ops, create_tour_from_X(X)
        
        return X, 0, ops, tour
        

    @staticmethod
    def double_bridge(X_i, free_nodes, fixed_edges, distance_matrix, CLs, tour, len_tour):
        # n = len(tour) -1
        X= np.copy(X_i)

        tour = create_tour_from_X(X)
        free_edges_current_tour, indeces = find_free_edges(free_nodes, tour, fixed_edges)
        # print("free edges are:")
        # print(free_edges_current_tour)

        max_value = max([8, len(free_edges_current_tour)//4])
        number_of_exhanges = np.random.randint(4, max_value)

        # randomly choose from the free edges four edges to operate the double bridge
        selected_edges = np.random.choice(free_edges_current_tour, number_of_exhanges, replace=False)
        selected_indices = [indeces[h] for h in selected_edges]
        # a, b, c, d = np.sort(selected_indices)
        # a_, b_, c_, d_ = selected_edges[np.argsort(selected_indices)]
        ops_ = 0
        new_cost = len_tour
        round_ = 0
        for i in np.sort(selected_indices):
            round_ += 1
            # print(f'\n round {round_}')
            X, gain, ops, tour = MLGreedy.two_cl(i, X,  tour, free_nodes, fixed_edges, distance_matrix, CLs)
            # print(gain, ops)
            new_cost -= gain
            ops_ += ops

        return X, create_tour_from_X(X), new_cost, ops_
        # solution = tour[:-1]
        # # Defining the segments of the solution ABCD
        # B = solution[a+1:b+1]
        # C = solution[b+1:c+1]
        # D = solution[c+1:d+1]
        # A = np.concatenate((solution[d+1-n:], solution[:a+1]))

        # Calculating the difference in cost between the current solution and the new solution
        # print(a_, b_, c_, d_)
        # print(a, b, c, d)
        # print(tour[a+1], tour[b+1], tour[c+1], tour[d-n])

        # old_cost = distance_matrix[a_, tour[a +1]] + distance_matrix[b_, tour[b+1]] + \
        #            distance_matrix[c_, tour[c+1]] + distance_matrix[d_, tour[d-n]]
        
        # new_cost_edges = distance_matrix[a_, tour[c+1]] + distance_matrix[b_, tour[d-n]] +\
        #                  distance_matrix[c_, tour[a+1]] + distance_matrix[d_, tour[b+1]]
        
        # gain = new_cost_edges - old_cost
        # # print(new_cost_edges, old_cost)
        # # print(f"Gain double bridge = {gain}")

        # # Adding the actual cost and the gain to get the new cost
        # new_cost = len_tour + gain

        # # # Defining the segments of the new solution ADCB
        # # new_solution = np.concatenate((A, D, C, B, [A[0]]))

        # # Adjust X matrix to the new tour
        # # print(X[a_], X[tour[a+1]], X[b_], X[tour[b+1]], X[c_], X[tour[c+1]], X[d_], X[tour[d-n]])

        # X[a_][np.where(X[a_] == tour[a+1])[0][0]] = tour[c+1]
        # X[toucalendar[c+1]][np.where(X[tour[c+1]] == c_)[0][0]] = a_
        
        # X[b_][np.where(X[b_] == tour[b+1])[0][0]] = tour[d-n]
        # X[tour[d-n]][np.where(X[tour[d-n]] == d_)[0][0]] = b_
        
        # X[c_][np.where(X[c_] == tour[c+1])[0][0]] = tour[a+1]
        # X[tour[a+1]][np.where(X[tour[a+1]] == a_)[0][0]] = c_
        
        # X[d_][np.where(X[d_] == tour[d-n])[0][0]] = tour[b+1]
        # X[tour[b+1]][np.where(X[tour[b+1]] == b_)[0][0]] = d_ 


        # print(X[a_], X[tour[a+1]], X[b_], X[tour[b+1]], X[c_], X[tour[c+1]], X[d_], X[tour[d-n]])

        # operate the double bridge on tour and X
    
    @staticmethod
    def save_tours_hashtable(tour, len_tour, seen_tours):
        tour1_to_save, tour2_to_save = standardize_tour(tour, both_tours=True)
        seen_tours[tour1_to_save] = len_tour
        seen_tours[tour2_to_save] = len_tour
        return seen_tours


    @staticmethod
    def run_ILS(X, distance_matrix, free_nodes, fixed_edges, CLs, type_2opt="reduced", opt_len=None):
        t0 = time.time()
        if type_2opt=="reduced":
            two_opt_fun = MLGreedy.two_opt_reduced_CL
        elif type_2opt=="free":
            two_opt_fun = MLGreedy.two_opt_free_CL
        else:
            two_opt_fun = MLGreedy.two_opt_complete_CL

        # copy the X variable to avoid changes on the initial value
        X_c = np.copy(X)
        tour_initial = create_tour_from_X(X_c) # it created the tour from the X matrix and computed its objective function
        initial_len = compute_tour_lenght(tour_initial, distance_matrix)
        tour_lens_list = [initial_len]
        current_len = initial_len
        current_tour = tour_initial
        ops_used = 0

        # initialize hashtable to keep track of the solutions already visited
        seen_tours = {}
        seen_tours = MLGreedy.save_tours_hashtable(current_tour, current_len, seen_tours)

        # Here it starts the first local search to improve the initial solution
        add_impro = 0
        repeat_optimization_r = 0
        while True:
                
            X_c, improvement, ops_plus, current_tour = two_opt_fun(X_c, free_nodes,
                                                                   fixed_edges, distance_matrix, 
                                                                   CLs, current_tour)
            ops_used += ops_plus 
            current_tour = current_tour[::-1]               
            current_len -= improvement
            add_impro += improvement
            repeat_optimization_r += 1
            repeat_optimization_r = repeat_optimization_r % 2
            if improvement==0 and repeat_optimization_r==0:
                break
            
            if repeat_optimization_r==0:
                standard_tour = standardize_tour(current_tour)
                if seen_tours.get(standard_tour, 0) == 0:
                    seen_tours = MLGreedy.save_tours_hashtable(current_tour, current_len, seen_tours)
                    # print(f"Process {mp.current_process().name},    dimension of the hashtable {len(seen_tours)}")
                    # print(f"Process {mp.current_process().name},    seen tours {seen_tours}")
                else:
                    break

        # print(f"Process {mp.current_process().name}, Initial tour len {current_len}")

        
        # update the variable used during the ILS 
        best_tour_so_far = current_tour
        best_len_so_far = current_len
        X_c_best = X_c
        tour_lens_list.append(best_len_so_far)
        temperature = initial_len * 10
        counter_temperature = 0
        repeat_temperature = 50
        probabilities = []
        count_iterations = 0
        avg_probs = 1
        continue_next_while = False

        # Here it starts the ILS
        while avg_probs>0.01:
            # Double Bridge
            X_proposal, tour_proposal, len_proposal,  ops_plus = MLGreedy.double_bridge(X_c, free_nodes, fixed_edges,
                                                                                        distance_matrix, CLs, 
                                                                                        current_tour, current_len)
            
            ops_used += ops_plus

            # checks if the perturbation is new
            standard_tour_prop = standardize_tour(tour_proposal)
            if seen_tours.get(standard_tour_prop, 0) == 0:
                seen_tours = MLGreedy.save_tours_hashtable(tour_proposal, len_proposal, seen_tours)
            else:
                # print(f"Process {mp.current_process().name}, perturbation already in hash table!")
                # print(standard_tour)
                continue

            # Here it starts the second local search to improve the solution after the double bridge
            add_impro = 0
            first_time_seen = False
            repeat_optimization_r = 0
            while True:
                    
                X_proposal, improvement, ops_plus, tour_proposal = two_opt_fun(X_proposal, free_nodes, fixed_edges,
                                                                               distance_matrix, CLs, tour_proposal)
                ops_used += ops_plus
                len_proposal -= improvement
                add_impro += improvement
                tour_proposal = tour_proposal[::-1]

                repeat_optimization_r += 1
                repeat_optimization_r = repeat_optimization_r % 2

                standard_tour_prop = standardize_tour(tour_proposal)
                if seen_tours.get(standard_tour_prop, 0) == 0:
                    seen_tours = MLGreedy.save_tours_hashtable(tour_proposal, len_proposal, seen_tours)
                    # print(f"Process {mp.current_process().name},  dimension of the hashtable {len(seen_tours)}")
                    first_time_seen = True
                else:
                    if repeat_optimization_r==0 and not first_time_seen:
                        continue_next_while = True
                        break
                
                if improvement == 0 and repeat_optimization_r==0:
                    break
            
            # print(f"Process avg probs = {avg_probs}, temperature = {temperature}")
            first_time_seen = False
            if continue_next_while and avg_probs>0.9:
                continue_next_while = False
                # print(f"Process {mp.current_process().name}   tour already seen continue!")
                continue
            continue_next_while = False

            # Here it checks if the new solution is accepted or not
            # if np.exp(-(len_proposal-current_len)/temperature) > np.random.uniform():
            if np.exp(np.clip(-(len_proposal-current_len)/temperature, -np.inf, np.inf)) > np.random.uniform():
                X_c = X_proposal
                current_tour = tour_proposal
                current_len = len_proposal
                tour_lens_list.append(current_len)
                # print(f"Process {mp.current_process().name},    current len accepted {current_len},"
                #       f" dimension of the hashtable {len(seen_tours)}, avg probs = {avg_probs}")
                    

            # Here it updates the temperature used for the ILS
            count_iterations += 1
            counter_temperature +=1
            # probabilities.append(np.exp(-(len_proposal-best_len_so_far)/temperature))
            probabilities.append(np.exp(np.clip(-(len_proposal-best_len_so_far)/temperature, -np.inf, np.inf)))

            if counter_temperature>repeat_temperature:
                counter_temperature = 0
                avg_probs = np.mean(probabilities)
                
                if avg_probs<0.7:
                    # repeat_temperature = 20
                    if avg_probs < 0.3:
                        # repeat_temperature = 30
                        if avg_probs < 0.1:
                            pass
                            # repeat_temperature = 40
                        else:
                            temperature *= 0.5
                    else:
                        temperature *= 0.75
                else:
                    temperature *= 0.9

                probabilities = []
            

            # Here it updates the best solution found so far
            if current_len < best_len_so_far:
                best_tour_so_far = current_tour
                best_len_so_far = current_len
                # assert best_len_so_far == compute_tour_lenght(best_tour_so_far, distance_matrix), \
                #                 f"Problem with the best tour {best_len_so_far} != {compute_tour_lenght(best_tour_so_far, distance_matrix)}"
                # print(f"Process {mp.current_process().name}")
                # print(f"\r$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ NEW BEST = {best_len_so_far}   "\
                #       f"iteration {count_iterations}  gap = {(best_len_so_far - opt_len)/opt_len * 100}   "
                #       f"temperature {temperature}    average prob {avg_probs}")
                # print(f"Process {mp.current_process().name},  dimension of the hashtable {len(seen_tours)}")
                    
                X_c_best = X_proposal
                
                # Here it checks if the solution is optimal in case it breaks the ILS
                if opt_len is not None:
                    if (current_len - opt_len )/opt_len*100< 0.0001:
                        # print(f"Process {__file__} breaking!")
                        break
                
        time_ils = time.time() - t0
        # print(f"\r###########FINAL RESULT ########## BEST LEN = {best_len_so_far}    last_iteration {count_iterations} "
        #       f" final gap =  {(best_len_so_far - opt_len)/opt_len * 100}   temperature {temperature}    average prob {avg_probs}  "
        #       f"ops used {ops_used}")
        return best_tour_so_far, X_c_best, tour_lens_list, time_ils, ops_used


    @staticmethod
    def improve_ILS_solution(X, X_intermediate, distance_matrix, CLs, opt_len, style="reduced"):
        fixed_edges = MLGreedy.get_fixed_edges(X_intermediate)
        free_nodes = MLGreedy.get_free_nodes(X_intermediate)


        # print(f"\n\nRunning approach REDUCED")
        # Run of the ILS using the reduced 2-opt
        best_tour_reduced, X_c_reduced, tour_lens_reduced, time_reduced, ops_reduced = MLGreedy.run_ILS(X, distance_matrix,
                                                                                                        free_nodes, fixed_edges,
                                                                                                        CLs, type_2opt=style,
                                                                                                        opt_len=opt_len)

        # print(f"\n\nRunning approach FREE")
        # Run of the ILS using the free 2-opt
        # best_tour_free, X_c_free, tour_lens_free, time_free, ops_free = MLGreedy.run_ILS(X, distance_matrix,
        #                                                                                 free_nodes, fixed_edges,
        #                                                                                 CLs, type_2opt="free",
        #                                                                                 opt_len=opt_len)


        # print(f"\n\nRunning approach COMPLETE")
        # Run of the ILS using the complete 2-opt
        # best_tour_complete, X_c_complete, tour_lens_complete, time_complete, ops_complete = MLGreedy.run_ILS(X, distance_matrix,
        #                                                                                                     free_nodes, fixed_edges,
        #                                                                                                     CLs, type_2opt="complete",
        #                                                                                                     opt_len=opt_len)


        data_to_return = {
            "tour Constructive": create_tour_from_X(X),
            "X Constructive": X, 

            f"tour ILS {style}": best_tour_reduced,
            f"X ILS {style}": X_c_reduced,
            f"list ILS {style} solutions": tour_lens_reduced,

            # "tour ILS free": best_tour_free,
            # "X ILS free": X_c_free,
            # "list ILS free solutions": tour_lens_free,
            # "tour ILS free": best_tour_reduced,
            # "X ILS free": X_c_reduced,
            # "list ILS free solutions": tour_lens_reduced,
            
            # "tour ILS complete": best_tour_complete,
            # "X ILS complete": X_c_complete,
            # "list ILS complete solutions": tour_lens_complete,

            
            f"Time ILS {style}": time_reduced,
            # "Time ILS free": time_free,
            # "Time ILS free": time_reduced,
            # "Time ILS complete": time_complete,

            f"Ops ILS {style}":ops_reduced,
            # "Ops ILS free":ops_free,
            # "Ops ILS free":ops_reduced,
            # "Ops ILS complete":ops_complete,

            "free_nodes": free_nodes,
            "fixed edges": fixed_edges,
        }
        return data_to_return


def check_feasibility(tour):
    n = len(tour) - 1
    return set(tour) == set(range(n))

def roll_the_tour(tour, X_c):
    shift = np.random.randint(1, len(tour)- 1)
    new_tour = np.roll(tour[:-1], shift)
    new_tour = np.append(new_tour, new_tour[0])
    new_X = np.roll(X_c, shift, axis=0)

    if np.random.randint(1) == 1:
        new_tour = new_tour[::-1]
        new_X = new_X[::-1]
    return new_tour, new_X


def find_free_edges(free_nodes, tour, fixed_edges):
    n = len(tour)
    free_edges_current_tour= []
    indeces_edges = {}
    for i, node_i in enumerate(tour[:-1]):
        if node_i in free_nodes:
            node_j = tour[:-1][i + 2 - n]
            # node_k = tour[:-1][i -1]
            if node_j in free_nodes:
                if (node_i, node_j) not in fixed_edges and (node_j, node_i) not in fixed_edges:
                    free_edges_current_tour.append(node_i)
                    indeces_edges[node_i] = i
                    # print(node_i, node_j)
                    # print(free_nodes)
    return free_edges_current_tour, indeces_edges

def standardize_tour(tour, both_tours=False):
    """
    it takes a tour as a list then finds the arg of the node 0 and then it rolls the tour.
    Finally, it returns two tours as a sting starting from 0 and ending with 0
    in both directions
    Arguments:
    """
    idx = np.argwhere(tour[:-1] == 0)[0][0]
    tour_to_return = np.roll(tour[:-1], -idx)
    tour_to_return = np.append(tour_to_return, tour_to_return[0])
    string_tour = ""
    for node in tour_to_return:
        string_tour += str(node) + "-"
    string_tour = string_tour[:-1]
    if both_tours:
        string_tour_2 = ""
        for node in tour_to_return[::-1]:
            string_tour_2 += str(node) + "-"
        string_tour_2 = string_tour_2[:-1]
        return string_tour, string_tour_2
    
    return string_tour