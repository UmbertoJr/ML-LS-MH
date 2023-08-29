import pandas as pd
import matplotlib.pyplot as plt


methods = ["Baseline", "Nearest Neighbour", "Linear", "Linear Underbalanced", 
           "Ensemble", "SVM", "Optimal Tour"]

for method in methods:
    df = pd.read_csv(f'./results/{method}_ls.csv', index_col=0)

    gap_columns = df.columns[0:4]
    for col in gap_columns:
        df[col] = df[col].str.replace("%", "").astype(float)

    time_columns = df.columns[4:8]
    time_df = pd.DataFrame()
    for col in time_columns:
        time_df[col] = df[col].str.replace(" sec", "").astype(float)
    
    ops_columns = df.columns[8:11]
    removed_columns = df.columns[11:]

    print("\n\n====================")
    print(f"Method: {method}")

    print("\nGap")
    print(df[gap_columns].mean())

    print("\nTime")
    print(time_df.sum())
    print(time_df.mean())

    print("\nOperations")
    # print(df[ops_columns].mean())
    print(df[ops_columns].sum())
    print(df[ops_columns].sum()/(df[ops_columns].sum().max()))

    print("\nRemoved edges from first phase")
    print(df[removed_columns].sum())
    print("\n====================")