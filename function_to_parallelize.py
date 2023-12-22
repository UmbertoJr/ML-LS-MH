import os
import json
import numpy as np
import multiprocessing as mp
from ML_greedy_v2 import MLGreedy
from drawer import plot_points_sol_intermediate
from random_reader import RandomInstancesGenerator
from utils import compute_difference_tour_length, compute_tour_lenght


def create_results(name_instance, improvement, ml_model, cl_method, n_points,
                   positions, distance_matrix, optimal_tour, shared_dict, style):
    
    # set priority for the process
    os.nice(0)

    # format_string = "{:^12}{:^18}{:^12}{:^15}{:^15}{:^15}{:^10}{:^18}{:^10}{:^10}"
    format_string = "{:^10}{:^15}{:^10}{:^15}{:^15}{:^15}{:^12}{:^12}{:^10}{:^10}"
    header = ["Problem", 
              "ML-model",
              "Gap ML-C", 
              f"Gap {improvement} {style}", 
              "Time ML-C", 
              f"Time {improvement} {style}", 
              f"Operations {improvement} {style}", 
              f"Best sol at operation {improvement} {style}",
              f"Removed {improvement} {style}", 
              f"First phase edges",
            ]
   
    # check if the experiment has already been done
    # read file f'./results/partial_results/{name_instance}_{improvement}_{ml_model}.json'
    # if the file exists, return the data in the file and skip the experiment
    # otherwise, run the experiment and save the results in the file
    # if os.path.exists(f'./results/partial_results/{name_instance}_{improvement}_{ml_model}_{style}.json'):
    #     with open(f'./results/partial_results/{name_instance}_{improvement}_{ml_model}_{style}.json') as fp:
    #         data = json.load(fp)
    #     shared_dict[name_instance] = data
    #     # print(f"Process {mp.current_process().name} is running {name_instance} "\
    #     #       f"with {improvement} {style} and {ml_model}...")   
    #     # print(f"shared_dict[{name_instance}] = {shared_dict[name_instance]}")
    #     values = [data[k][0] for k in header]
    #     print(format_string.format(*values))
    #     return
   
    data = {h: [] for h in header}
    # filtered_head = header[:]
    # print(format_string.format(*filtered_head))

    # print(f"Process {mp.current_process().name} is running {name_instance} with {improvement} {style} and {ml_model}...")

    opt_tour = np.append(optimal_tour, optimal_tour[0])
    opt_len = compute_tour_lenght(opt_tour,distance_matrix)

    experiments_results = MLGreedy.run(n_points, positions, 
                                       distance_matrix, optimal_tour, 
                                       cl_method=cl_method, 
                                       ml_model=ml_model, 
                                       opt_len=opt_len,
                                       improvement_type=improvement, 
                                       style=style,
                                       name_instance=name_instance)
    

    # mlg_tour = create_tour_from_X(experiments_results["tour"])
    mlg_tour = experiments_results["tour Constructive"]

    # print(mlg_tour)
    # print(f"optimal len to achieve = {opt_len}")
    # print(experiments_results[f"tour {improvement} {style}"])
    # print(compute_difference_tour_length(opt_tour, experiments_results[f"tour {improvement} {style}"], distance_matrix)*100)

    # plot_points_sol_intermediate(positions, experiments_results["X Constructive"], 
    #                              experiments_results["X Intermediate"], mlg_tour)
    # plot_points_sol_intermediate(positions, experiments_results[f"X {improvement} {style}"], 
    #                              experiments_results["X Intermediate"],
    #                              experiments_results[f"tour {improvement} {style}"])
    
    delta = compute_difference_tour_length(opt_tour, mlg_tour,
                                            distance_matrix)
    delta_imp_reduced = compute_difference_tour_length(opt_tour, 
                                                       experiments_results[f"tour {improvement} {style}"], 
                                                    distance_matrix)
    
    # print(experiments_results[f"tour {improvement} {style}""])
    
    count_removed_fixed_edges_reduced = 0
    count_first_phase_edges = int(len(experiments_results["fixed edges"])/2)
    for edge in experiments_results["fixed edges"]:
        a, b = edge[0], edge[1]

        if b not in experiments_results[f"X {improvement} {style}"][a]:
            count_removed_fixed_edges_reduced += 1
        if a not in experiments_results[f"X {improvement} {style}"][b]:
            count_removed_fixed_edges_reduced += 1
        

    row = [
        name_instance,
        str(ml_model),
        f'{delta * 100:.3f} %', 
        f'{delta_imp_reduced * 100:.3f} %',
        
        f'{experiments_results["Time ML-C"]:.3f} sec', 
        f'{experiments_results[f"Time {improvement} {style}"]:.3f} sec', 
        
        experiments_results[f"Ops {improvement} {style}"], 
        experiments_results[f"Best sol at opertion {improvement} {style}"],
        
        count_removed_fixed_edges_reduced,
        count_first_phase_edges,
        ]
    
    print(format_string.format(*row))

    for i, h in enumerate(header):
        data[h].append(row[i])

    shared_dict[name_instance] = data

    if not os.path.exists(f'./results/partial_results'):
        os.makedirs(f'./results/partial_results')

    # save the experiment results as a json file in the "./results/partial_results" folder
    with open(f'./results/partial_results/{name_instance}_{improvement}_{ml_model}_{style}.json', 'w') as fp:
        json.dump(data, fp, indent=4)