import time
import numpy as np
from utils import create_tour_from_X
from candidate_list import CandidateList
from machine_learning_models import MLAdd
from utils import compute_tour_lenght
import multiprocessing as mp


# TODO:
# - [ ] cancellare parti di codice inutili
# - [ ] implementare un perturbation operator efficace

TO_PRINT = False


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

        # create CL for each vertex the nn cl is used for the ML models, while the other cl is the POPMUSIC
        # one used for the 2-opt and the ILS
        candidate_list_nn = CandidateList.compute(positions, distance_matrix, cl_method)
        candidate_list, alpha_list = CandidateList.read_CL_from_file(name_instance)
        if len(candidate_list) == 0:
            candidate_list = candidate_list_nn
        
        
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
                    nodes = np.concatenate(([i], candidate_list_nn[i][:limit]))
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

        ml_tour = create_tour_from_X(X)

        if improvement_type == "2-Opt":
            results_to_return = MLGreedy.improve_solution(X, 
                                                          X_intermediate,
                                                          distance_matrix, 
                                                          candidate_list,
                                                          style=style,
                                                          opt_len=opt_len,
                                                          alpha_list=alpha_list)
        elif improvement_type == "2-Opt-CL":
            results_to_return = MLGreedy.improve_solution(X, 
                                                          X_intermediate, 
                                                          distance_matrix, 
                                                          candidate_list, 
                                                          with_CL=True,
                                                          style=style,
                                                          opt_len=opt_len,
                                                          alpha_list=alpha_list)
        elif improvement_type == "ILS":
            results_to_return = MLGreedy.improve_ILS_solution(X, 
                                                              X_intermediate, 
                                                              distance_matrix, 
                                                              candidate_list, 
                                                              opt_len, 
                                                              style=style,
                                                              alpha_list=alpha_list)

        results_to_return["tour NO simple"] = ml_tour
        results_to_return["X NO simple"] = X        
        results_to_return["Time ML-C"] = time_mlg
        results_to_return["X Intermediate"] = X_intermediate

        return results_to_return
        
    @staticmethod
    def get_free_nodes(X, tour=None):
        """
        This function takes X intermediate and returns the nodes that has at least one connection free
        """
        """
        This function takes X intermediate and returns the nodes that has at least one connection free
        """
        free_nodes = np.array([i for i in range(len(X)) if (X[i] == -1).any()])
        if tour is not None:
            # save position of the free nodes in the tour
            free_nodes_position = {node: int(np.argwhere(tour == node)[0][0]) for node in free_nodes}
        return free_nodes, free_nodes_position

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
    def improve_solution(X, X_intermediate, 
                         distance_matrix, CLs,
                         with_CL = False, 
                         style="complete", 
                         opt_len=0,
                         alpha_list={}):
        
        
        X_c = np.copy(X)
        tour_ = create_tour_from_X(X_c)
        len_tour = compute_tour_lenght(tour_, distance_matrix)
        fixed_edges = MLGreedy.get_fixed_edges(X_intermediate)
        free_nodes, position_free_nodes = MLGreedy.get_free_nodes(X_intermediate, tour_)
        tabu_list = {}
        
        
        if style == "reduced":
            if with_CL:
                improvement_function = MLGreedy.two_opt_reduced_CL
            else:
                improvement_function = MLGreedy.two_opt_reduced

        else:
            assert False, "style not implemented"
        

        #  qui si lancia il 2-opt
        t0 = time.time()
        count_ = 0
        ops_ = 0

        while True:
            X_c, tour_, improvement, ops_, tabu_list, position_free_nodes = improvement_function(X_c, tour_,
                                                                                                free_nodes, 
                                                                                                position_free_nodes,
                                                                                                fixed_edges,
                                                                                                distance_matrix, 
                                                                                                CLs,
                                                                                                tabu_list=tabu_list)
            count_ += ops_
            len_tour -= improvement
            # print(f'Process {mp.current_process().name} improvement = {improvement},'
            #       f'len_tour = {len_tour}, gap = {(len_tour - opt_len) / opt_len * 100:.3f} %')
            
            if improvement == 0:
                time_ = time.time() - t0
                break
        
        # tour_X = create_tour_from_X(X_c)
        # len_tour_computed = compute_tour_lenght(tour_X, distance_matrix)
        # len_tour_ = compute_tour_lenght(tour_, distance_matrix)
        # assert len_tour == len_tour_computed, f"len_tour = {len_tour}, len_tour_computed = {len_tour_computed}"
        # assert len_tour_ == len_tour_computed, f"len_tour_ = {len_tour_}, len_tour_computed = {len_tour_computed}"
        # assert len_tour_ == len_tour, f"len_tour_ = {len_tour_}, len_tour = {len_tour}"

        data_to_return = {
            "tour Constructive": create_tour_from_X(X),
            "X Constructive": X, 

            f"tour 2-Opt{['', '-CL'][with_CL]} {style}": tour_,
            f"X 2-Opt{['', '-CL'][with_CL]} {style}": X_c,
            
            f"Time 2-Opt{['', '-CL'][with_CL]} {style}": time_,
      
            f"Ops 2-Opt{['', '-CL'][with_CL]} {style}":count_,
            
            "free_nodes": free_nodes,
            "fixed edges": fixed_edges,
        }
        return data_to_return
                
    @staticmethod
    def two_opt_reduced(X, tour, free_nodes, position_free_nodes, fixed_edges, distance_matrix, CLs, tabu_list= False):
        ops = 0
        n = len(tour)
        for node_ip in free_nodes:
            for andamento in [1 - n, -1]:
                node_in = tour[position_free_nodes[node_ip] + andamento]
                if node_in in free_nodes and (node_ip, node_in) not in fixed_edges:
                    
                    for node_jp in free_nodes:
                        if node_jp not in [node_ip, node_in]:
                            node_jn = tour[position_free_nodes[node_jp] + andamento]
                            if node_jn not in [node_in, node_ip]:
                                if node_jn in free_nodes and (node_jp, node_jn) not in fixed_edges:
                                    if not tabu_list or check_if_tabu(node_ip, node_in, node_jp, node_jn, tabu_list):
                                            
                                        old_cost = distance_matrix[node_ip][node_in] + distance_matrix[node_jp][node_jn]
                                        new_cost = distance_matrix[node_ip][node_jp] + distance_matrix[node_in][node_jn]
                                        ops += 1
                                        if old_cost - new_cost > 0:
                                            if TO_PRINT:
                                                print("\nentrato!")
                                                print(f"node_ip = {node_ip}, node_in = {node_in}")
                                                print(f"andamento = {andamento}, n= {n}")
                                                print()
                                                print(f"node_jp = {node_jp}, node_jn = {node_jn}")
                                                print()
                                                print(f"distance_matrix[node_ip][node_in] = {distance_matrix[node_ip][node_in]}")
                                                print(f"distance_matrix[node_jp][node_jn] = {distance_matrix[node_jp][node_jn]}")
                                                print(f"distance_matrix[node_ip][node_jp] = {distance_matrix[node_ip][node_jp]}")
                                                print(f"distance_matrix[node_in][node_jn] = {distance_matrix[node_in][node_jn]}")
                                                print(f"old_cost = {old_cost}, new_cost = {new_cost}")
                                                print()
                                
                                                print(node_ip, node_in, node_jp, node_jn)
                                                print(f"improvement = {old_cost - new_cost}")
                                                print()

                                            X[node_ip][np.where(X[node_ip] == node_in)[0][0]] = node_jp
                                            X[node_in][np.where(X[node_in] == node_ip)[0][0]] = node_jn
                                            X[node_jp][np.where(X[node_jp] == node_jn)[0][0]] = node_ip
                                            X[node_jn][np.where(X[node_jn] == node_jp)[0][0]] = node_in
                                            
                                            
                                            position_minimal = min(position_free_nodes[node_ip], position_free_nodes[node_jp])
                                            position_maximal = max(position_free_nodes[node_ip], position_free_nodes[node_jp])

                                            if TO_PRINT:
                                                print(f"position_minimal = {position_minimal}, position_maximal = {position_maximal}")
                                                print(f"tour = {tour}")


                                            if andamento == -1:

                                                # update the position of the free nodes in the tour
                                                for iter, node_id in enumerate(range(position_minimal, position_maximal)):
                                                    node = tour[node_id]
                                                    if node in free_nodes:
                                                        # print(f"node = {node}")
                                                        # print(f"previous postion node = {position_free_nodes[node]}")
                                                        position_free_nodes[node] = position_minimal + (position_maximal - position_minimal - iter - 1)
                                                        # print("new position of the node")
                                                        # print(position_free_nodes[node])

                                                # print(f"tour[position_minimal:position_maximal] = {tour[position_minimal:position_maximal]}")
                                                tour[position_minimal:position_maximal] = \
                                                    np.flip(tour[position_minimal:position_maximal], axis=0)
                                                    
                                                # print(np.argwhere(tour == node)[0][0])
                                                # print()
                                                
                                            else:
                                                if position_maximal == n-1:

                                                    # update the position of the free nodes in the tour
                                                    for iter, node_id in enumerate(range(position_minimal, n)):
                                                        node = tour[node_id]
                                                        if node in free_nodes:
                                                            # print(f"node = {node}")
                                                            # print(f"previous postion node = {position_free_nodes[node]}")
                                                            position_free_nodes[node] = position_minimal + (position_maximal - position_minimal - iter)
                                                            # print("new position of the node")
                                                            # print(position_free_nodes[node])
            
                                                    # print(f"tour[position_minimal:] = {tour[position_minimal:]}")
                                                    tour[position_minimal+andamento:] = \
                                                        np.flip(tour[position_minimal+andamento:], axis=0)

                                                    # print(np.argwhere(tour == node)[0][0])
                                                    # print()

                                                else:    
                                                    # update the position of the free nodes in the tour
                                                    for iter, node_id in enumerate(range(position_minimal+1, position_maximal+1)):
                                                        node = tour[node_id]
                                                        if node in free_nodes:
                                                            # print(f"node = {node}")
                                                            # print(f"previous postion node = {position_free_nodes[node]}")
                                                            position_free_nodes[node] = position_minimal + (position_maximal - position_minimal - iter)
                                                            # print("new position of the node")
                                                            # print(position_free_nodes[node])

                                            
                                                    # print(f"tour[position_minimal+andamento:position_maximal+andament] = {tour[position_minimal+andamento:position_maximal+andamento]}")
                                                    tour[position_minimal+andamento:position_maximal+andamento] = \
                                                        np.flip(tour[position_minimal+andamento:position_maximal+andamento], axis=0)

                                            # check if tabu list is a active (if it is active, it is a dictionary)
                                            if type(tabu_list) == dict:
                                                # insert the edge in the tabu list
                                                insert_tabu(node_ip, node_in, node_jp, node_jn, tabu_list)
                                            

                                            return X, tour, old_cost - new_cost, ops, tabu_list, position_free_nodes
                                        
                                    else:
                                        evaporate_tabu(tabu_list)
            
        return X, tour, 0, ops, tabu_list, position_free_nodes
    

    @staticmethod
    def two_opt_reduced_CL(X, tour, free_nodes, position_free_nodes, fixed_edges, distance_matrix, CLs, tabu_list= False):
        ops = 0
        n = len(tour)
        for node_ip in free_nodes:
            for andamento in [1 - n, -1]:
                node_in = tour[position_free_nodes[node_ip] + andamento]
                if node_in in free_nodes and (node_ip, node_in) not in fixed_edges:
                    possible_nodes_jp = CLs[node_ip] + CLs[node_in]
                    for node_jp in possible_nodes_jp:
                        if node_jp not in [node_ip, node_in]:
                            if node_jp in free_nodes:
                                node_jn = tour[position_free_nodes[node_jp] + andamento]
                                if node_jn not in [node_in, node_ip, node_jp]:
                                    if node_jn in free_nodes and (node_jp, node_jn) not in fixed_edges:
                                        if not tabu_list or check_if_tabu(node_ip, node_in, node_jp, node_jn, tabu_list):
                                            old_cost = distance_matrix[node_ip][node_in] + distance_matrix[node_jp][node_jn]
                                            new_cost = distance_matrix[node_ip][node_jp] + distance_matrix[node_in][node_jn]
                                            ops += 1
                                            if old_cost - new_cost > 0:
                                                if TO_PRINT:        
                                                    print("\nLocal Search!")
                                                    print(f"node_ip = {node_ip}, node_in = {node_in}")
                                                    print(f"andamento = {andamento}, n= {n}")
                                                    print()
                                                    print(f"node_jp = {node_jp}, node_jn = {node_jn}")
                                                    print()
                                                    print(f"distance_matrix[node_ip][node_in] = {distance_matrix[node_ip][node_in]}")
                                                    print(f"distance_matrix[node_jp][node_jn] = {distance_matrix[node_jp][node_jn]}")
                                                    print(f"distance_matrix[node_ip][node_jp] = {distance_matrix[node_ip][node_jp]}")
                                                    print(f"distance_matrix[node_in][node_jn] = {distance_matrix[node_in][node_jn]}")
                                                    print(f"old_cost = {old_cost}, new_cost = {new_cost}")
                                                    print()
                                    
                                                    print(node_ip, node_in, node_jp, node_jn)
                                                    print(f"improvement = {old_cost - new_cost}")
                                                    print()

                                                    print(f"X[{node_ip}] = {X[node_ip]}, X[{node_in}] = {X[node_in]}")
                                                    print(f"X[{node_jp}] = {X[node_jp]}, X[{node_jn}] = {X[node_jn]}")
                                                    print(f"==========================")

                                                X[node_ip][np.where(X[node_ip] == node_in)[0][0]] = node_jp
                                                X[node_in][np.where(X[node_in] == node_ip)[0][0]] = node_jn
                                                X[node_jp][np.where(X[node_jp] == node_jn)[0][0]] = node_ip
                                                X[node_jn][np.where(X[node_jn] == node_jp)[0][0]] = node_in
                                                
                                                if TO_PRINT:
                                                    print(f"X[{node_ip}] = {X[node_ip]}, X[{node_in}] = {X[node_in]}")
                                                    print(f"X[{node_jp}] = {X[node_jp]}, X[{node_jn}] = {X[node_jn]}")
                                                    print()

                                                    print()
                                                    print(f"position_free_nodes[node_ip] = {position_free_nodes[node_ip]}")
                                                    print(f"position_free_nodes[node_in] = {position_free_nodes[node_in]}") 
                                                    print(f"position_free_nodes[node_jp] = {position_free_nodes[node_jp]}")
                                                    print(f"position_free_nodes[node_jn] = {position_free_nodes[node_jn]}")


                                                    positions = [position_free_nodes[node_ip], position_free_nodes[node_in], 
                                                                 position_free_nodes[node_jp], position_free_nodes[node_jn]]
                                                    
                                                    print(f"positions = {positions}")
                                                    print(f"np.abs(position_free_nodes[node_in] - position_free_nodes[node_jp]) = {np.abs(position_free_nodes[node_in] - position_free_nodes[node_jp])}")
                                                    print(f"position_free_nodes[node_ip] - position_free_nodes[node_jn] = {np.abs(position_free_nodes[node_ip] - position_free_nodes[node_jn])}")
                                                

                                                to_flip = np.argmin([np.abs(position_free_nodes[node_in] - position_free_nodes[node_jp]), 
                                                np.abs(position_free_nodes[node_ip] - position_free_nodes[node_jn])])

                                                p_min = [position_free_nodes[node_in], position_free_nodes[node_ip]][to_flip]
                                                p_max = [position_free_nodes[node_jp], position_free_nodes[node_jn]][to_flip]

                                                position_minimal = min(p_min, p_max)
                                                position_maximal = max(p_min, p_max)

                                                if TO_PRINT:    
                                                    print(f"to_flip = {to_flip}")
                                                    print(f"position_minimal = {position_minimal}, position_maximal = {position_maximal}")
                                                    print(f"tour = {tour}")

                                                # update the position of the free nodes in the tour
                                                for iter, node_id in enumerate(range(position_minimal, position_maximal + 1)):
                                                    node = tour[node_id]
                                                    if node in free_nodes:
                                                        if TO_PRINT:
                                                            print(f"node = {node}")
                                                            print(f"previous postion node = {position_free_nodes[node]}")
                                                        position_free_nodes[node] = position_minimal + (position_maximal - position_minimal - iter)
                                                        if TO_PRINT:
                                                            print("new position of the node")
                                                            print(position_free_nodes[node])
                                                

                                                if TO_PRINT:
                                                    print(f"tour[position_minimal:position_maximal] = {tour[position_minimal:position_maximal + 1]}")
                                                tour[position_minimal:position_maximal + 1] = \
                                                    np.flip(tour[position_minimal:position_maximal + 1], axis=0)
                                                
                                                if TO_PRINT:  
                                                    print(f"tour[position_minimal:position_maximal] = {tour[position_minimal:position_maximal + 1]}")
                                                    print(np.argwhere(tour == node)[0][0])

                                                # check if tabu list is a active (if it is active, it is a dictionary)
                                                if type(tabu_list) == dict:
                                                    # insert the edge in the tabu list
                                                    insert_tabu(node_ip, node_in, node_jp, node_jn, tabu_list)
                                                

                                                return X, tour, old_cost - new_cost, ops, tabu_list, position_free_nodes
                                            
                                        else:
                                            evaporate_tabu(tabu_list)                                                
            
        return X, tour, 0, ops, tabu_list, position_free_nodes
    

    @staticmethod
    def two_cl(i, X, tour, free_nodes, position_free_nodes, fixed_edges, distance_matrix, CLs):
        n = len(tour)
        node_in = tour[i]
        node_ip = tour[i -1]
        ops = 0
        if node_ip in free_nodes and node_in in free_nodes and (node_ip, node_in) not in fixed_edges:    
            closest_nodes = CLs[node_ip] + CLs[node_in]
            for node_jn in np.random.permutation(closest_nodes):    
                if node_jn not in [node_ip, node_in] and node_jn in free_nodes and node_jn not in [node_ip, node_in, 
                                                                                                   tour[position_free_nodes[node_ip] -1], 
                                                                                                   tour[position_free_nodes[node_in] - n + 1]]:
                    j = position_free_nodes[node_jn]
                    node_jp = tour[j - 1]
                    if node_jp in free_nodes and (node_jp, node_jn) not in fixed_edges and node_jp not in [node_ip, node_in,
                                                                                                           tour[position_free_nodes[node_ip] -1], 
                                                                                                           tour[position_free_nodes[node_in] - n + 1]]:
                        ops += 1
                        old_cost = distance_matrix[node_ip][node_in] + distance_matrix[node_jp][node_jn]
                        new_cost = distance_matrix[node_ip][node_jp] + distance_matrix[node_in][node_jn]
                        
                        if TO_PRINT:
                            print("\nPerturbate!")
                            print(f"node_ip = {node_ip}, node_in = {node_in}")
                            print()
                            print(f"node_jp = {node_jp}, node_jn = {node_jn}")
                            print()
                            print(f"distance_matrix[node_ip][node_in] = {distance_matrix[node_ip][node_in]}")
                            print(f"distance_matrix[node_jp][node_jn] = {distance_matrix[node_jp][node_jn]}")
                            print(f"distance_matrix[node_ip][node_jp] = {distance_matrix[node_ip][node_jp]}")
                            print(f"distance_matrix[node_in][node_jn] = {distance_matrix[node_in][node_jn]}")
                            print(f"old_cost = {old_cost}, new_cost = {new_cost}")
                            print()
            
                            print(node_ip, node_in, node_jp, node_jn)
                            # print(f"improvement = {old_cost - new_cost}")
                            # print()

                            print(f"X[{node_ip}] = {X[node_ip]}, X[{node_in}] = {X[node_in]}")
                            print(f"X[{node_jp}] = {X[node_jp]}, X[{node_jn}] = {X[node_jn]}")
                            print(f"==========================")

                        X[node_ip][np.where(X[node_ip] == node_in)[0][0]] = node_jp
                        X[node_in][np.where(X[node_in] == node_ip)[0][0]] = node_jn
                        X[node_jp][np.where(X[node_jp] == node_jn)[0][0]] = node_ip
                        X[node_jn][np.where(X[node_jn] == node_jp)[0][0]] = node_in
                        
                        if TO_PRINT:
                            print(f"X[{node_ip}] = {X[node_ip]}, X[{node_in}] = {X[node_in]}")
                            print(f"X[{node_jp}] = {X[node_jp]}, X[{node_jn}] = {X[node_jn]}")
                        

                            print()
                            print(f"position_free_nodes[node_ip] = {position_free_nodes[node_ip]}")
                            print(f"position_free_nodes[node_in] = {position_free_nodes[node_in]}") 
                            print(f"position_free_nodes[node_jp] = {position_free_nodes[node_jp]}")
                            print(f"position_free_nodes[node_jn] = {position_free_nodes[node_jn]}")


                            positions = [position_free_nodes[node_ip], position_free_nodes[node_in], 
                                        position_free_nodes[node_jp], position_free_nodes[node_jn]]
                            print(f"positions = {positions}")

                            print(f"np.abs(position_free_nodes[node_in] - position_free_nodes[node_jp]) = {np.abs(position_free_nodes[node_in] - position_free_nodes[node_jp])}")
                            print(f"position_free_nodes[node_ip] - position_free_nodes[node_jn] = {np.abs(position_free_nodes[node_ip] - position_free_nodes[node_jn])}")
                        to_flip = np.argmin([np.abs(position_free_nodes[node_in] - position_free_nodes[node_jp]), 
                                             np.abs(position_free_nodes[node_ip] - position_free_nodes[node_jn])])
                        

                        p_min = [position_free_nodes[node_in], position_free_nodes[node_ip]][to_flip]
                        p_max = [position_free_nodes[node_jp], position_free_nodes[node_jn]][to_flip]


                        position_minimal = min(p_min, p_max)
                        position_maximal = max(p_min, p_max)

                        if TO_PRINT:
                            print(f"p_min = {p_min}, p_max = {p_max}")
                            print(f"to_flip = {to_flip}")

                            print()
                            print(f"position_minimal = {position_minimal}, position_maximal = {position_maximal}")
                            print(f"tour[position_minimal:position_maximal] = {tour[position_minimal:position_maximal + 1 - n]}")
                            print(f"tour = {tour}")

                        # update the position of the free nodes in the tour
                        for iter, node_id in enumerate(range(position_minimal, position_maximal + 1)):
                            node = tour[node_id]
                            if node in free_nodes:
                                if TO_PRINT:
                                    print(f"node = {node}")
                                    print(f"previous postion node = {position_free_nodes[node]}")

                                position_free_nodes[node] = position_minimal + (position_maximal - position_minimal - iter)

                                if TO_PRINT:
                                    print("new position of the node")
                                    print(position_free_nodes[node])
                                

                        if TO_PRINT:
                            print(f"tour[position_minimal:position_maximal] = {tour[position_minimal:position_maximal + 1 - n]}")

                        tour[position_minimal:position_maximal + 1] = \
                            np.flip(tour[position_minimal:position_maximal + 1], axis=0)
                        
                        if TO_PRINT:
                            print(f"tour[position_minimal:position_maximal] = {tour[position_minimal:position_maximal + 1 - n]}")
                            print(np.argwhere(tour == node)[0][0])

                        return X, old_cost - new_cost, ops, tour, position_free_nodes
        
        return X, 0, ops, tour, position_free_nodes
        

    @staticmethod
    def perturbation(X_i, tour, len_tour, free_nodes, position_free_nodes, fixed_edges, distance_matrix, CLs, tabu_list):
        # TODO: implementare un perturbation operator efficace
        # TODO: implementare un roll del tour
        # TODO: usare i valori di alpha per decidere quali vertici perturbare
        
        # n = len(tour) -1
        X= np.copy(X_i)
        tour_proposal = np.copy(tour)
        len_proposal = np.copy(len_tour)

        free_edges_current_tour, indeces = find_free_edges(free_nodes, tour_proposal, fixed_edges)
        # print("free edges are:")
        # print(free_edges_current_tour)
        
        max_value = max([8, len(free_edges_current_tour)//4])
        number_of_exhanges = np.random.randint(4, max_value)

        # randomly choose from the free edges four edges to operate the double bridge
        selected_edges = np.random.choice(free_edges_current_tour, number_of_exhanges, replace=False)
        selected_indices = [indeces[h] for h in selected_edges]
        
        ops_ = 0
        round_ = 0
        # print(compute_tour_lenght(tour, distance_matrix))
        for i in np.sort(selected_indices):
            round_ += 1
            
            X, gain, ops, tour_proposal,\
                 position_free_nodes = MLGreedy.two_cl(i, X,  tour_proposal, 
                                                       free_nodes, position_free_nodes, 
                                                       fixed_edges, distance_matrix, CLs)
            
            len_proposal -= gain
            # print(f"round = {round_}, gain = {gain}, len_proposal = {len_proposal}")
            ops_ += ops
        
        
        assert len_proposal == compute_tour_lenght(tour_proposal, distance_matrix), \
                            f"Problem with the proposal tour {len_proposal} != {compute_tour_lenght(tour_proposal, distance_matrix)}"
        return X, tour_proposal, len_proposal, position_free_nodes, ops_


    @staticmethod
    def local_search_call(improvement_function, X_c, tour_, len_tour, free_nodes, position_free_nodes, fixed_edges, distance_matrix, CLs, tabu_list= False): 

        count_ = 0
        ops_ = 0

        while True:
            X_c, tour_, improvement, ops_,\
                  tabu_list, position_free_nodes = improvement_function(X_c, tour_,
                                                                        free_nodes, 
                                                                        position_free_nodes,
                                                                        fixed_edges,
                                                                        distance_matrix, 
                                                                        CLs,
                                                                        tabu_list=tabu_list)
            count_ += ops_
            len_tour -= improvement
            # print(f'Process {mp.current_process().name} improvement = {improvement},'
            #       f'len_tour = {len_tour}, gap = {(len_tour - opt_len) / opt_len * 100:.3f} %')
            
            if improvement == 0:
                break
        
        assert len_tour == compute_tour_lenght(tour_, distance_matrix), \
                            f"Problem with the proposal tour {len_tour} != {compute_tour_lenght(tour_, distance_matrix)}"
        return X_c, tour_, len_tour, count_, tabu_list, position_free_nodes

    @staticmethod
    def run_ILS(X, X_intermediate, distance_matrix, CLs, opt_len=None, alpha_list={}):
        
        t0 = time.time()

        # select the improvement function
        two_opt_fun = MLGreedy.two_opt_reduced_CL
        
        # restrain the number of iterations to a bilion
        n = len(X)
        ops_used = 0
        total_iterations_available = n*n*10

        # copy the initial solution and create tour
        X_c = np.copy(X)
        tour_initial = create_tour_from_X(X_c)
        initial_len = compute_tour_lenght(tour_initial, distance_matrix)

        # finde the fixed edges and the free nodes
        fixed_edges = MLGreedy.get_fixed_edges(X_intermediate)
        free_nodes, position_free_nodes = MLGreedy.get_free_nodes(X_intermediate, tour_initial)

        # initialize the tabu list to keep track of the actions recently performed
        tabu_list = {}

        # set useful variable to take trac of the ILS iterations
        temperature = initial_len * 10
        counter_temperature = 0
        repeat_temperature = 50
        probabilities = []
        count_iterations = 0
        avg_probs = 1
        continue_next_while = False
        
        tour_lens_list = [initial_len]
        current_len = initial_len
        current_tour = tour_initial

        best_tour_so_far = current_tour
        best_len_so_far = current_len
        X_c_best = X_c
        tour_lens_list.append(best_len_so_far)
        
        # Local Search Call on the initial solution
        X_c, current_tour, current_len, ops_used,\
              tabu_list, position_free_nodes = MLGreedy.local_search_call(two_opt_fun, X_c, 
                                                                          tour_initial, initial_len,
                                                                          free_nodes, position_free_nodes,
                                                                          fixed_edges, distance_matrix,
                                                                          CLs, tabu_list=tabu_list)
        
        # Here it updates the best solution found so far
        if current_len < best_len_so_far:
            best_tour_so_far = current_tour
            best_len_so_far = current_len
            assert best_len_so_far == compute_tour_lenght(best_tour_so_far, distance_matrix), \
                            f"Problem with the best tour {best_len_so_far} != {compute_tour_lenght(best_tour_so_far, distance_matrix)}"

            if TO_PRINT:
                print(f"Process {mp.current_process().name}")
                print(f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ NEW BEST = {best_len_so_far}   "\
                    f"iteration {count_iterations}  gap = {(best_len_so_far - opt_len)/opt_len * 100}   "
                    f"temperature {temperature}    average prob {avg_probs}   ops used {ops_used}")
                # print(f"Initial solution improvement initial_len - best_so_far = {initial_len - best_len_so_far}")
                print()            
        

        # Here it starts the ILS
        while avg_probs>0.7:
            free_nodes, position_free_nodes = MLGreedy.get_free_nodes(X_intermediate, current_tour)

            # Perturbate the current solution with Double Bridge
            X_proposal, tour_proposal, len_proposal, position_free_nodes,\
                    ops_plus = MLGreedy.perturbation(X_c,
                                                     current_tour, 
                                                     current_len,
                                                     free_nodes, 
                                                     position_free_nodes,
                                                     fixed_edges,
                                                     distance_matrix, 
                                                     CLs, 
                                                     tabu_list=tabu_list)
            
            ops_used += ops_plus

            

            # COMMENT: If the perturbation does not give improvements often Let's see how can we improve it
            
            # Local Search Call on the perturbated solution using the Tabu List
            X_proposal, tour_proposal, len_proposal, ops_plus,\
                    tabu_list, position_free_nodes = MLGreedy.local_search_call(two_opt_fun, X_proposal , 
                                                                                tour_proposal, len_proposal,
                                                                                free_nodes, position_free_nodes,
                                                                                fixed_edges, distance_matrix,
                                                                                CLs, tabu_list=tabu_list)

            
            # It checks if the new proposal solution is accepted or not
            if np.exp(np.clip(-(len_proposal-current_len)/temperature, -np.inf, np.inf)) > np.random.uniform():
                X_c = X_proposal
                current_tour = tour_proposal
                current_len = len_proposal
                tour_lens_list.append(current_len)

                # Then, it check if the new solution is better than the best solution found so far
                if current_len < best_len_so_far:
                    X_c_best = X_proposal
                    best_tour_so_far = current_tour
                    best_len_so_far = current_len
                    assert best_len_so_far == compute_tour_lenght(best_tour_so_far, distance_matrix), \
                                    f"Problem with the best tour {best_len_so_far} != {compute_tour_lenght(best_tour_so_far, distance_matrix)}"
                    if TO_PRINT:
                        print(f"Process {mp.current_process().name}")
                        print(f"\r$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ NEW BEST = {best_len_so_far}   "\
                            f"iteration {count_iterations}  gap = {(best_len_so_far - opt_len)/opt_len * 100}   "
                            f"temperature {temperature}    average prob {avg_probs}   ops used {ops_used}")
                        print()

                    # Here it checks if the solution is optimal in case it breaks the ILS
                    if opt_len is not None:
                        if (current_len - opt_len )/opt_len*100< 0.1:
                            break
            
            
            # ops_used += ops_plus
            # if ops_used>total_iterations_available:
            #     break
            
            # It updates the temperature used for the ILS
            count_iterations += 1
            counter_temperature +=1
            probabilities.append(np.exp(np.clip(-(len_proposal-best_len_so_far)/temperature, -np.inf, np.inf)))

            if counter_temperature>repeat_temperature:
                counter_temperature = 0
                avg_probs = np.mean(probabilities)
                if TO_PRINT:
                    print(f"avg_probs = {avg_probs}")
                
                if avg_probs<0.7:
                    break
                    # repeat_temperature = 20
                    # if avg_probs < 0.3:
                    #     # repeat_temperature = 30
                    #     if avg_probs < 0.1:
                    #         pass
                    #         # repeat_temperature = 40
                    #     else:
                    #         temperature *= 0.5
                    # else:
                    #     temperature *= 0.75
                else:
                    temperature *= 0.9

                probabilities = []
            
                
                
        time_ils = time.time() - t0
        if TO_PRINT:
            print(f"\r###########FINAL RESULT ########## BEST LEN = {best_len_so_far}    last_iteration {count_iterations} "
                f" final gap =  {(best_len_so_far - opt_len)/opt_len * 100}   temperature {temperature}    average prob {avg_probs}  "
                f"ops used {ops_used}")
        return best_tour_so_far, X_c_best, tour_lens_list, time_ils, free_nodes, fixed_edges, ops_used


    @staticmethod
    def improve_ILS_solution(X, X_intermediate, distance_matrix, CLs, opt_len, style="reduced", alpha_list={}):

        # print(f"\n\nRunning approach REDUCED")
        # Run of the ILS using the reduced 2-opt
        best_tour_reduced, X_c_reduced,\
              tour_lens_reduced, time_reduced, free_nodes, fixed_edges,\
                ops_reduced = MLGreedy.run_ILS(X, X_intermediate, distance_matrix,
                                               CLs, opt_len=opt_len, alpha_list=alpha_list)


        data_to_return = {
            "tour Constructive": create_tour_from_X(X),
            "X Constructive": X, 

            f"tour ILS {style}": best_tour_reduced,
            f"X ILS {style}": X_c_reduced,
            f"list ILS {style} solutions": tour_lens_reduced,
            
            f"Time ILS {style}": time_reduced,
            
            f"Ops ILS {style}":ops_reduced,
            
            "free_nodes": free_nodes,
            "fixed edges": fixed_edges,
        }
        return data_to_return


def check_if_tabu(i, j, k, l, tabu):
    first = min(i, j)
    second = max(i, j)

    third = min(k, l)
    fourth = max(k, l)

    # check if the first or the third edge is smaller
    if first < third:
        if ((first, second), (third, fourth)) in tabu:
            return True
        else:
            return False
        
    elif first > third:
        if ((third, fourth), (first, second)) in tabu:
            return True
        else:
            return False
        
def insert_tabu(i, j, k, l, tabu):
    first = min(i, j)
    second = max(i, j)

    third = min(k, l)
    fourth = max(k, l)

    # check if the first or the third edge is smaller
    if first < third:
        tabu[((first, second), (third, fourth))] = 10
        
    elif first > third:
        tabu[((third, fourth), (first, second))] = 10

    return tabu

def evaporate_tabu(tabu):
    if len(tabu) == 0:
        return tabu
    keys_to_delete = []
    for key in tabu:
        tabu[key] -= 1
        if tabu[key] == 0:
            keys_to_delete.append(key)
    for key in keys_to_delete:
        del tabu[key]
    return tabu



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
