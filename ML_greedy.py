import time
import numpy as np
from utils import create_tour_from_X
from candidate_list import CandidateList
from machine_learning_models import MLAdd
from utils import compute_tour_lenght

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
    def run(n, positions, distance_matrix, optimal_tour, cl_method=CandidateList.Method.NearestNeighbour,
            ml_model=MLAdd.MLModel.NearestNeighbour, limit=15, opt_len=None):
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

        # for l in L_P select the extreme vertices i, j of l
        for L_P_pos, [i, j] in enumerate(L_P.T):
            # if i and j have less than two connections each in X
            if np.sum(X[i] > -1) < 2 and np.sum(X[j] > -1) < 2:

                # if l do not creates a inner-loop
                if not MLGreedy.inner_loop(X.copy(), [i, j]):
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
                    if ML_add(distance=L_P_distances[L_P_pos], distance_vector=dists, solution_vector=edges_in_sol, in_opt=ret):
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


        # X_improved, time_2opt = MLGreedy.improve_solution(X, X_intermediate, distance_matrix, candidate_list)
        # results_to_return = MLGreedy.improve_solution(X, X_intermediate, distance_matrix, candidate_list)
        
        results_to_return = MLGreedy.improve_ILS_solution(X, X_intermediate, distance_matrix, candidate_list, opt_len)


        
        results_to_return["Time ML-C"] = time_mlg
        results_to_return["X Intermediate"] = X_intermediate
        return results_to_return
        # X_ILS, time_ILS, operation_ILS, calls_LS, 
        # return X, X_intermediate, X_improved, time_mlg, time_2opt
        # return X, X_intermediate, X, time_mlg, 0
        # return X, X_intermediate, X_improved, time_mlg, time_2opt

    @staticmethod
    def get_free_nodes(X):
        """
        This function takes X intermediate and returns the nodes that has at least one connection free
        """
        free_nodes = np.array([i for i in range(len(X)) if (X[i] == -1).any()])
        return free_nodes

    @staticmethod
    def get_fixed_edges(X):
        """
        This function takes X_intermediate and returns in a list all the edges connetcted during first phase
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
    def improve_solution(X, X_intermediate, distance_matrix, CLs):
        fixed_edges = MLGreedy.get_fixed_edges(X_intermediate)
        free_nodes = MLGreedy.get_free_nodes(X_intermediate)
        
        #  qui si lancia il 2-opt ridotto
        t0 = time.time()
        X_c_reduced = np.copy(X)
        tour_reduced = create_tour_from_X(X_c_reduced)
        count_reduced = 0
        ops_reduced = 0

        while True:
            # tour_reduced, X_c_reduced = roll_the_tour(tour_reduced, X_c_reduced)
            X_improved, improvement, ops_plus = MLGreedy.two_opt_reduced(X_c_reduced, free_nodes, fixed_edges,
                                                                         distance_matrix, CLs, tour_reduced)
            
            X_c_reduced = X_improved
            tour_reduced = create_tour_from_X(X_c_reduced)
            assert check_feasibility(tour_reduced), "Problem the tour is not feasible"
            ops_reduced += ops_plus
            if improvement == 0:
                time_reduced = time.time() - t0
                # print(improvement)
                # print(ops_reduced)
                break
            else:
                # print(improvement)
                # print(ops_reduced)
                count_reduced += 1
        

        #  qui si lancia il 2-opt free
        t0 = time.time()
        X_c_free = np.copy(X)
        tour_free = create_tour_from_X(X_c_free)
        count_free = 0
        ops_free = 0
        while True:
            # tour_free = roll_the_tour(tour_free)
            # X_c_free = X_c_free[::-1]
            X_improved, improvement, ops_plus = MLGreedy.two_opt_free(X_c_free, free_nodes, fixed_edges, 
                                                                      distance_matrix, CLs, tour_free)
            X_c_free = X_improved
            tour_free = create_tour_from_X(X_c_free)
            tour_free = tour_free[::-1]
            assert check_feasibility(tour_free), "Problem the tour is not feasible"
            ops_free += ops_plus
            if improvement == 0 and count_free > 1:
                time_free = time.time() - t0
                break
            else:
                count_free += 1

        
        #  qui si lancia il 2-opt complete
        t0 = time.time()
        X_c_complete = np.copy(X) 
        tour_complete = create_tour_from_X(X_c_complete)
        count_complete = 0
        ops_complete = 0
        while True:
            # tour_complete = roll_the_tour(tour_complete)
            X_improved, improvement, ops_plus = MLGreedy.two_opt_complete(X_c_complete, free_nodes, fixed_edges, 
                                                                          distance_matrix, CLs, tour_complete)
            
            X_c_complete = X_improved
            tour_complete = create_tour_from_X(X_improved)
            assert check_feasibility(tour_complete), "Problem the tour is not feasible"
            ops_complete += ops_plus
            if improvement == 0:
                time_complete = time.time() - t0
                break
            else:
                count_complete += 1


        data_to_return = {
            "tour Constructive": create_tour_from_X(X),
            "X Constructive": X, 

            "tour 2-Opt reduced": tour_reduced,
            "X 2-Opt reduced": X_c_reduced,

            "tour 2-Opt free": tour_free, 
            "X 2-Opt free": X_c_free,

            "tour 2-Opt complete": tour_complete,
            "X 2-Opt complete": X_c_complete,
            
            "Time 2-Opt reduced": time_reduced,
            "Time 2-Opt free": time_free,
            "Time 2-Opt complete": time_complete,

            "Ops 2-Opt reduced":ops_reduced,
            "Ops 2-Opt free":ops_free,
            "Ops 2-Opt complete":ops_complete,

            "free_nodes": free_nodes,
            "fixed edges": fixed_edges,
        }
        return data_to_return
        
        # # qui si lancia il 2-opt completo
        # while True:
        #     X_improved_complete, improvement = MLGreedy.two_opt(X, free_nodes, fixed_edges, distance_matrix, CLs)
        #     if improvement == 0:
        #         time_complete = time.time - t0
        #         break
                
        # changed_edges_LS = MLGreedy.find_changed(fixed_edges, X_improved_complete)

        # data_to_return = {
        #     "sol after reduced 2-Opt": X_improved, 
        #     "time reduced 2-Opt": time_simple, 
        #     "": operations_LS, 
        #     "": changed_edges_LS, 
        #     "": X_improved_complete, 
        #     "": time_complete
        # }
        # return 

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
    def double_bridge(X_i, free_nodes, fixed_edges, distance_matrix, CLs, tour, len_tour):
        n = len(tour) -1
        X= np.copy(X_i)

        # tour = create_tour_from_X(X)
        free_edges_current_tour, indeces = find_free_edges(free_nodes, tour, fixed_edges)
        # print("free edges are:")
        # print(free_edges_current_tour)

        # randomly choose from the free edges four edges to operate the double bridge
        selected_edges = np.random.choice(free_edges_current_tour, 4, replace=False)
        selected_indices = [indeces[h] for h in selected_edges]
        a, b, c, d = np.sort(selected_indices)
        a_, b_, c_, d_ = selected_edges[np.argsort(selected_indices)]

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

        old_cost = distance_matrix[a_, tour[a +1]] + distance_matrix[b_, tour[b+1]] + \
                   distance_matrix[c_, tour[c+1]] + distance_matrix[d_, tour[d-n]]
        
        new_cost_edges = distance_matrix[a_, tour[c+1]] + distance_matrix[b_, tour[d-n]] +\
                         distance_matrix[c_, tour[a+1]] + distance_matrix[d_, tour[b+1]]
        
        gain = new_cost_edges - old_cost
        # print(new_cost_edges, old_cost)
        # print(f"Gain double bridge = {gain}")

        # Adding the actual cost and the gain to get the new cost
        new_cost = len_tour + gain

        # # Defining the segments of the new solution ADCB
        # new_solution = np.concatenate((A, D, C, B, [A[0]]))

        # Adjust X matrix to the new tour
        # print(X[a_], X[tour[a+1]], X[b_], X[tour[b+1]], X[c_], X[tour[c+1]], X[d_], X[tour[d-n]])

        X[a_][np.where(X[a_] == tour[a+1])[0][0]] = tour[c+1]
        X[tour[c+1]][np.where(X[tour[c+1]] == c_)[0][0]] = a_
        
        X[b_][np.where(X[b_] == tour[b+1])[0][0]] = tour[d-n]
        X[tour[d-n]][np.where(X[tour[d-n]] == d_)[0][0]] = b_
        
        X[c_][np.where(X[c_] == tour[c+1])[0][0]] = tour[a+1]
        X[tour[a+1]][np.where(X[tour[a+1]] == a_)[0][0]] = c_
        
        X[d_][np.where(X[d_] == tour[d-n])[0][0]] = tour[b+1]
        X[tour[b+1]][np.where(X[tour[b+1]] == b_)[0][0]] = d_ 


        # print(X[a_], X[tour[a+1]], X[b_], X[tour[b+1]], X[c_], X[tour[c+1]], X[d_], X[tour[d-n]])

        # operate the double bridge on tour and X
        return X, create_tour_from_X(X), new_cost, n
    

    @staticmethod
    def improve_ILS_solution(X, X_intermediate, distance_matrix, CLs, opt_len):
        fixed_edges = MLGreedy.get_fixed_edges(X_intermediate)
        free_nodes = MLGreedy.get_free_nodes(X_intermediate)
        
        #  qui si lancia ILS ridotto
        t0 = time.time()

        X_c_reduced = np.copy(X)
        tour_reduced = create_tour_from_X(X_c_reduced)
        initial_len = compute_tour_lenght(tour_reduced, distance_matrix)
        tour_lens_reduced = [initial_len]
        new_len = initial_len
        count_reduced = 0
        ops_reduced = 0
        add_impro = 0
        while True:
                
            X_c_reduced, improvement, ops_plus, tour_reduced = MLGreedy.two_opt_reduced(X_c_reduced, free_nodes, fixed_edges,
                                                                                        distance_matrix, CLs, tour_reduced)
            ops_reduced += ops_plus                
            new_len -= improvement
            add_impro += improvement
            if improvement==0:
                break
        # print(f"improvement 2-opt = {add_impro}")
        # print(f"new len={new_len}")
        best_tour_so_far_reduced = create_tour_from_X(X_c_reduced)
        current_len = compute_tour_lenght(best_tour_so_far_reduced, distance_matrix)
        assert current_len == new_len, f"{current_len} and {new_len} should be equal"
        best_len_so_far = current_len
        tour_lens_reduced.append(best_len_so_far)
        temperature = initial_len * 10
        print(f'initial temperature = {temperature}')
        # print(tour_reduced)
        print(f"initial len = {current_len}")
        count_temperature = 0
        probabilities = []
        while temperature>0.001:
            # print("\nInitial Tour is:")
            # print(tour_reduced)
            X_improved, new_tour, new_len, ops_plus = MLGreedy.double_bridge(X_c_reduced, free_nodes, fixed_edges,
                                                                              distance_matrix, CLs, 
                                                                              tour_reduced, current_len)
            
            ops_reduced += ops_plus
            # print("new tour is:")
            # print(new_tour)
            add_impro = 0
            while True:
                    
                X_improved, improvement, ops_plus, new_tour = MLGreedy.two_opt_reduced(X_improved, free_nodes, fixed_edges,
                                                                                       distance_matrix, CLs, new_tour)
                ops_reduced += ops_plus
                new_len = new_len - improvement
                add_impro += improvement
                if improvement == 0:
                    break
            # print(f"improvement 2-opt = {add_impro}")
            # print(f"new len={new_len}")
            count_reduced += 1
            count_temperature +=1
            if count_temperature>100:
                temperature *= 0.5
                count_temperature = 0
                print(f"############ temperature = {temperature}")
                print(f"############ average probability = {np.mean(probabilities)}\n")
                probabilities = []
            # print(f"probability = {np.exp(-(new_len-best_len_so_far)/temperature)}")
            probabilities.append(np.exp(-(new_len-current_len)/temperature))
            if new_len < best_len_so_far:
                tour_ = new_tour[::-1]
                assert check_feasibility(tour_), f"Problem the tour is not feasible {tour_}"
                
                best_tour_so_far_reduced = new_tour
                best_len_so_far = new_len
                X_c_reduced = X_improved
                tour_reduced = new_tour
                current_len = new_len
                print(f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ NEW BEST = {current_len}   iteration {count_reduced}  opt len = {opt_len}")
                tour_lens_reduced.append(current_len)
                if current_len - opt_len == 0:
                    print("breaking!")
                    break
                # print(improvement)
                # print(ops_reduced)
                # break
            elif np.exp(-(new_len-best_len_so_far)/temperature) > np.random.uniform():
                X_c_reduced = X_improved
                tour_reduced = new_tour
                current_len = new_len
                tour_lens_reduced.append(current_len)
                # print(improvement)
                # print(ops_reduced)
            # tour_reduced = create_tour_from_X(X_improved)
            # X_c_reduced = X_improved
            # tour_reduced = create_tour_from_X(X_c_reduced)
            # assert check_feasibility(tour_reduced), "Problem the tour is not feasible"
        

        time_reduced = time.time() - t0

        print(tour_lens_reduced)
        print(f"{count_reduced} iteration ILS ")
        print(f"best result = {min(tour_lens_reduced)}")
        #  qui si lancia il 2-opt free
        t0 = time.time()
        X_c_free = np.copy(X)
        tour_free = create_tour_from_X(X_c_free)
        count_free = 0
        ops_free = 0
        # while True:
        #     # tour_free = roll_the_tour(tour_free)
        #     # X_c_free = X_c_free[::-1]
        #     X_improved, improvement, ops_plus = MLGreedy.two_opt_free(X_c_free, free_nodes, fixed_edges, 
        #                                                               distance_matrix, CLs, tour_free)
        #     X_c_free = X_improved
        #     tour_free = create_tour_from_X(X_c_free)
        #     tour_free = tour_free[::-1]
        #     assert check_feasibility(tour_free), "Problem the tour is not feasible"
        #     ops_free += ops_plus
        #     if improvement == 0 and count_free > 1:
        #         time_free = time.time() - t0
        #         break
        #     else:
        #         count_free += 1

        


        #  qui si lancia il 2-opt complete
        t0 = time.time()
        X_c_complete = np.copy(X) 
        tour_complete = create_tour_from_X(X_c_complete)
        count_complete = 0
        ops_complete = 0
        # while True:
        #     # tour_complete = roll_the_tour(tour_complete)
        #     X_improved, improvement, ops_plus = MLGreedy.two_opt_complete(X_c_complete, free_nodes, fixed_edges, 
        #                                                                   distance_matrix, CLs, tour_complete)
            
        #     X_c_complete = X_improved
        #     tour_complete = create_tour_from_X(X_improved)
        #     assert check_feasibility(tour_complete), "Problem the tour is not feasible"
        #     ops_complete += ops_plus
        #     if improvement == 0:
        #         time_complete = time.time() - t0
        #         break
        #     else:
        #         count_complete += 1


        data_to_return = {
            "tour Constructive": create_tour_from_X(X),
            "X Constructive": X, 

            "tour ILS reduced": best_tour_so_far_reduced,
            "X ILS reduced": X_c_reduced,
            "list ILS reduced solutions": tour_lens_reduced,

            "tour ILS free": tour_free, 
            "X ILS free": X_c_free,
            "list ILS free solutions": tour_lens_reduced,

            "tour ILS complete": tour_complete,
            "X ILS complete": X_c_complete,
            "list ILS complete solutions": tour_lens_reduced,
            
            "Time ILS reduced": time_reduced,
            # "Time ILS free": time_free,
            # "Time ILS complete": time_complete,

            "Ops ILS reduced":ops_reduced,
            "Ops ILS free":ops_free,
            "Ops ILS complete":ops_complete,

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