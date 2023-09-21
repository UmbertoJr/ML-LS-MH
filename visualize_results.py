import pandas as pd
import matplotlib.pyplot as plt

# improvement = "2-Opt-CL"
improvement = "NO"
style = "simple"
# improvement = "ls"

ops_tot = {
    "Baseline": 1479935115,
    "Nearest Neighbour": 1048230780,
    "Linear": 931919595,
    "SVM": 807041716,
    "Ensemble": 1412224215,
    "Optimal Tour": 632318661,
}

# methods = ["Baseline", "Nearest Neighbour", "Linear", "SVM", 
#            "Ensemble", "Optimal Tour"]
methods = ["Optimal Tour"]

for method in methods:
    df = pd.read_csv(f'./results/{method}_{improvement}_{style}.csv') #, index_col=0)
    # print(df.head())
    colonna_n = df["Problem"].str.extract(r"(\d+)").astype(int)
    print(colonna_n.sum())



    # ml_model_col = 1
    # # gap_columns = df.columns[0+ml_model_col:4+ml_model_col]
    # gap_columns = df.columns[1:3]
    # for col in gap_columns:
    #     df[col] = df[col].str.replace("%", "").astype(float)

    # # time_columns = df.columns[4+ml_model_col:8+ml_model_col]
    # time_columns = df.columns[3:5]
    # time_df = pd.DataFrame()
    # for col in time_columns:
    #     time_df[col] = df[col].str.replace(" sec", "").astype(float)
    
    # # ops_columns = df.columns[8+ml_model_col:11+ml_model_col]
    # ops_columns = df.columns[5]
    # # removed_columns = df.columns[11+ml_model_col:]
    # removed_columns = df.columns[6]

    # first_column = df.columns[7]

    # print("\n\n====================")
    # print(f"Method: {method}")

    # print("\nGap")
    # print(df[gap_columns].mean())

    # # print("\nTime")
    # # print(time_df.sum())
    # # print(time_df.mean())

    # # print("\nOperations")
    # # # # print(df[ops_columns].mean())
    # # print(df[ops_columns].sum())
    # # print(df[ops_columns].sum()/(df[ops_columns].sum().max()))
    # # print(df[ops_columns].sum()/ops_tot[method])

    # # print("\nRemoved edges")
    # # print(df[removed_columns].sum())

    # print("\nFirst phase edges")
    # print(df[first_column].sum())
    # print("\n====================")