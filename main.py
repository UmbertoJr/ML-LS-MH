import os
import shutil
import numpy as np
import pandas as pd
from tsplib_reader import ReadTSPlib
from candidate_list import CandidateList
from machine_learning_models import MLAdd
import multiprocessing as mp
from function_to_parallelize import create_results


cl_method = CandidateList.Method.NearestNeighbour
ml_model = MLAdd.MLModel.RN
improvement = "ILS"  # 2-Opt, 2-Opt-CL or ILS
style = "reduced"  # reduced, free or complete


if __name__ == "__main__":
    os.nice(0)

    # Numbert of prcess available
    num_processes = mp.cpu_count() - 3
    # num_processes = 1
    print(f"Number of processes available: {num_processes}")
        
    # delete all the files in the partial results folder
    # if os.path.exists('./results/partial_results'):
    #     for filename in os.listdir(f'./results/partial_results'):
    #         file_path = os.path.join(f'./results/partial_results', filename)
    #         try:
    #             if os.path.isfile(file_path) or os.path.islink(file_path):
    #                 os.unlink(file_path)
    #             elif os.path.isdir(file_path):
    #                 shutil.rmtree(file_path)
    #         except Exception as e:
    #             print('Failed to delete %s. Reason: %s' % (file_path, e))

    
    # reader = RandomInstancesGenerator() MLAdd.MLModel.OptimalTour
    for ml_model in [MLAdd.MLModel.NearestNeighbour, MLAdd.MLModel.Baseline,
                     MLAdd.MLModel.Linear, MLAdd.MLModel.LinearUnderbalance,
                     MLAdd.MLModel.SVM, MLAdd.MLModel.Ensemble,
                     MLAdd.MLModel.OptimalTour, MLAdd.MLModel.RN]:
    # for style in ["reduced", "free", "complete"]:
    # for style in ["reduced"]:
        print('\n\n')
        print(f'--------------------------------------------------')
        print()
        print(f'Candidate List Method: {cl_method}')
        print(f'ML Model: {ml_model}')
        print(f'Improvement: {improvement}')
        print(f'Style: {style}')
        print()
        print(f'--------------------------------------------------')
        print()

        manager = mp.Manager()
        shared_dict = manager.dict()
        reader = ReadTSPlib()
        args = []
        
        for instance in reader.instances_generator():
            n_points, positions, distance_matrix, name, optimal_tour = instance
            
            # check if dimension is less than 700 otherwise skip
            if n_points > 500:
                continue

            args.append((name, improvement, ml_model, 
                         cl_method, n_points, positions, 
                         distance_matrix, optimal_tour, 
                         shared_dict, style))
            

            # break

        print(f"Arguments to pass to the parallelized function: {len(args)}\n\n")



        format_string = "{:^15}{:^15}{:^15}{:^20}{:^20}{:^20}{:^20}{:^20}{:^20}"
        header = ["Problem", 
                "ML-model",
                "Gap ML-C", 
                f"Gap {improvement} {style}", 
                "Time ML-C", 
                f"Time {improvement} {style}", 
                f"Operations {improvement} {style}", 
                f"Removed {improvement} {style}", 
                f"First phase edges",
                ]
        data = {h: [] for h in header}
        print(format_string.format(*header))

        # Now the parallelization using multiprocessing starmap
        with mp.Pool(processes=num_processes) as pool:
            pool.starmap(create_results, args)

        # take the shared_dict and create the dataframe
        data = {h: [] for h in header}
        for instance in shared_dict.keys():
            for h in header:
                data[h].append(shared_dict[instance][h][0])

        df = pd.DataFrame(data)
        # df.loc['Avg'] = df.mean()
        # df.loc['Tot'] = df.sum() 
        df.to_csv(f"./results/{ml_model}_{improvement}_{style}.csv", index=False)
        # print(df)

        # print('\n\n')
        # print(f'--------------------------------------------------')
        # print()
        # print(f'Candidate List Method: {cl_method}')
        # print(f'ML Model: {ml_model}')
        # print()
        # print(f'--------------------------------------------------')
        # print()

        # spinner = Halo(text='Loading', spinner='dots')
        # format_string = "{:^15}{:^15}{:^20}{:^20}{:^20}"
        # header = ["Problem", 
        #         "Gap ML-C", f"Gap {improvement} reduce", f"Gap {improvement} free", 
        #         #   f"Gap {improvement} complete",  
        #         "Time ML-C", f"Time {improvement} reduce", f"Time {improvement} free", 
        #         #   f"Time {improvement} complete",  
        #         f"Operations {improvement} reduce", f"Operations {improvement} free", 
        #         #   f"Operations {improvement} complete", 
        #         f"Removed {improvement} reduce", f"Removed {improvement} free", 
        #         #   f"Removed {improvement} complete", 
        #         ]
        # data = {h: [] for h in header}
        # filtered_head = header[:5]
        # print(format_string.format(*filtered_head))

        # for instance in reader.instances_generator():
        #     n_points, positions, distance_matrix, name, optimal_tour = instance


        #     # print(f"\n INSTANCE: {name}")
        #     # if n_points < 500:
        #     #     # break
        #     #     continue
        #     opt_tour = np.append(optimal_tour, optimal_tour[0])
        #     opt_len = compute_tour_lenght(opt_tour,distance_matrix)

        #     spinner.start()

        #     # X_g, X_intermediate, X_improved, time_mlg, time_2opt
        #     experiments_results = MLGreedy.run(n_points, positions, distance_matrix, 
        #                                     optimal_tour, cl_method=cl_method, 
        #                                     ml_model=ml_model, opt_len=opt_len, improvement_type=improvement)
            

        #     # mlg_tour = create_tour_from_X(experiments_results["tour"])
        #     mlg_tour = experiments_results["tour Constructive"]

        #     # print(mlg_tour)
        #     # print(f"optimal len to achieve = {opt_len}")
        #     # print(experiments_results[f"tour {improvement} reduced"])
        #     # print(compute_difference_tour_length(opt_tour, experiments_results[f"tour {improvement} reduced"], distance_matrix)*100)

        #     # plot_points_sol_intermediate(positions, experiments_results["X Constructive"], 
        #     #                              experiments_results["X Intermediate"], mlg_tour)
        #     # plot_points_sol_intermediate(positions, experiments_results[f"X {improvement} reduced"], 
        #     #                              experiments_results["X Intermediate"],
        #     #                              experiments_results[f"tour {improvement} reduced"])
            
        #     # plot_points_sol_intermediate(positions, experiments_results[f"X {improvement} free"], 
        #     #                              experiments_results["X Intermediate"],
        #     #                              experiments_results[f"tour {improvement} free"])
            
        #     # plot_points_sol_intermediate(positions, experiments_results[f"X {improvement} complete"], 
        #     #                              experiments_results["X Intermediate"],
        #     #                              experiments_results[f"tour {improvement} complete"])
        #     # plot_points_sol_intermediate(positions, X_improved, X_intermediate)

        #     # delta_improved = compute_difference_tour_length(opt_tour, create_tour_from_X(X_improved), distance_matrix)
        #     delta = compute_difference_tour_length(opt_tour, mlg_tour,
        #                                             distance_matrix)
        #     delta_imp_reduced = compute_difference_tour_length(opt_tour, experiments_results[f"tour {improvement} reduced"], 
        #                                                     distance_matrix)
        #     delta_imp_free = compute_difference_tour_length(opt_tour, experiments_results[f"tour {improvement} free"], distance_matrix)
        #     # delta_imp_complete = compute_difference_tour_length(opt_tour, experiments_results[f"tour {improvement} complete"], 
        #     #                                                     distance_matrix)

        #     # print(experiments_results[f"tour {improvement} reduced"])
        #     # print(experiments_results[f"tour {improvement} free"])
        #     # print(experiments_results[f"tour {improvement} complete"])
        #     # print(delta_imp_reduced, delta_imp_free, delta_imp_complete)

        #     count_removed_fixed_edges_reduced = 0
        #     count_removed_fixed_edges_free = 0
        #     count_removed_fixed_edges_complete = 0
        #     for edge in experiments_results["fixed edges"]:
        #         a, b = edge[0], edge[1]

        #         if b not in experiments_results[f"X {improvement} reduced"][a]:
        #             count_removed_fixed_edges_reduced += 1
        #         if a not in experiments_results[f"X {improvement} reduced"][b]:
        #             count_removed_fixed_edges_reduced += 1
                
        #         if b not in experiments_results[f"X {improvement} free"][a]:
        #             count_removed_fixed_edges_free += 1
        #         if a not in experiments_results[f"X {improvement} free"][b]:
        #             count_removed_fixed_edges_free += 1
                
        #         # if b not in experiments_results[f"X {improvement} complete"][a]:
        #         #     count_removed_fixed_edges_complete += 1
        #         # if a not in experiments_results[f"X {improvement} complete"][b]:
        #         #     count_removed_fixed_edges_complete += 1


        #     spinner.stop()

        #     row = [
        #         name,

        #         f'{delta * 100:.3f} %', 
        #         f'{delta_imp_reduced * 100:.3f} %',
        #         f'{delta_imp_free * 100:.3f} %', 
        #         # f'{delta_imp_complete * 100:.3f} %',
                
        #         f'{experiments_results["Time ML-C"]:.3f} sec', 
        #         f'{experiments_results[f"Time {improvement} reduced"]:.3f} sec', 
        #         f'{experiments_results[f"Time {improvement} free"]:.3f} sec', 
        #         # f'{experiments_results[f"Time {improvement} complete"]:.3f} sec', 
                
        #         experiments_results[f"Ops {improvement} reduced"], 
        #         experiments_results[f"Ops {improvement} free"], 
        #         # experiments_results[f"Ops {improvement} complete"], 

        #         count_removed_fixed_edges_reduced,
        #         count_removed_fixed_edges_free,
        #         # count_removed_fixed_edges_complete,
        #         ]
            
        #     row_filtered = row[:5]
        #     print(format_string.format(*row_filtered))
        #     # print(row)

        #     for i, h in enumerate(header):
        #         data[h].append(row[i])
            # time.sleep(1)

        # df = pd.DataFrame(data)
        # # df.loc['Avg'] = df.mean()
        # # df.loc['Tot'] = df.sum() 
        # df.to_csv(f"./results/{ml_model}_{improvement}.csv", index=False)
        # print(df)