# This is the file for processing data.
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

import math
if __name__ == "__main__":
    train_data_path = 'fifa_train.csv'
    validation_data_path = 'fifa_validation.csv'
    test_data_path = 'fifa_22test.csv'
    original_data_path_17 = 'players_17.csv'
    original_data_path_18 = 'players_18.csv'
    original_data_path_19 = 'players_19.csv'
    original_data_path_20 = 'players_20.csv'
    original_data_path_21 = 'players_21.csv'
    original_data_path_22 = 'players_22.csv'
    

    # dt_original_17 = pd.read_csv(original_data_path_17, low_memory = False)
    # dt_original_18 = pd.read_csv(original_data_path_18, low_memory = False)
    # dt_original_19 = pd.read_csv(original_data_path_19, low_memory = False)
    # dt_original_20 = pd.read_csv(original_data_path_20, low_memory = False)
    # dt_original_21 = pd.read_csv(original_data_path_21, low_memory = False)
    dt_original_22 = pd.read_csv(original_data_path_22, low_memory = False)
    # dt_original_17.head()
    # dt_original_18.head()
    # dt_original_19.head()
    # dt_original_20.head()
    # dt_original_21.head()
    dt_original_22.head()
    dataset1 = pd.concat([dt_original_22], ignore_index=True)                  # get the values from the csv file
    print(dataset1.shape)

    position_data = dataset1["player_positions"].str.split(",").str[0]
    dataset1["player_positions"] = position_data
    rating = dataset1.iloc[:, 5]
    value = dataset1.iloc[:, 7]
    potential = dataset1.iloc[:, 6]
    age = dataset1.iloc[:, 9]
    information = dataset1.iloc[:, [0, 4, 11, 12, 13, 17, 22, 27, 28, 29, 30, 31]]
    attribute = dataset1.iloc[:, 43:77]

    age_p = np.zeros((19239, 1))

    position_mapping = {
        "ST": 1,
        "CF": 2,
        "LW": 3,
        "RW": 4,
        "CAM": 5,
        "LM": 6,
        "RM": 7,
        "CM": 8,
        "CDM": 9,
        "LWB": 10,
        "RWB": 11,
        "LB": 12,
        "RB": 13,
        "CB": 14,
        "GK": 15
    }
    fallback_value = 0
    dataset1["player_positions"] = dataset1["player_positions"].map(position_mapping).fillna(fallback_value)
    
    dataset1.iloc[:, 13].fillna(0, inplace=True)
    dataset1.iloc[:, 13] = pd.factorize(dataset1.iloc[:, 13])[0]
    dataset1.iloc[:, 17].fillna("Unknown", inplace=True)
    dataset1.iloc[:, 17] = pd.factorize(dataset1.iloc[:, 17])[0]
    dataset1.iloc[:, 27] = pd.factorize(dataset1.iloc[:, 27])[0]
    dataset1.iloc[:, 31] = pd.factorize(dataset1.iloc[:, 31])[0]

    for i in range(19239):
        tmp = age[i]
        age_p[i] = tmp * 12 - 192

    dataset1.iloc[:, 9] = age_p[:, 0].astype(float)
    
    dataset2 = np.zeros((19239, 48))            # create a second dataset for storing the processed data
    dataset2[:, :14] = dataset1.iloc[:, [0, 4, 6, 9, 11, 12, 13, 17, 22, 27, 28, 29, 30, 31]].values.astype(int)
    dataset2[:, 1] = dataset2[:, 1].astype(int)
    dataset2[:, 14:48] = dataset1.iloc[:, 43:77].values.astype(int)
    # dataset2[:, [48, 49]] = dataset1.iloc[:, [5, 7]].values.astype(int)

    # row_ranges = [(0, 17595), (17596, 35549), (35550, 53634), (53635, 72117), (72118, 91061), (91062, 110300)]
    # version = 17       
    # for (start, end) in row_ranges:
    #     dataset2[start:end, 0] = version
    #     version += 1

    # print(version)

    sorting_key = dataset2[:, 0].astype(int)
    sorted_id = np.argsort(sorting_key)
    dataset2_sorted = dataset2[sorted_id]

    # test_data = pd.DataFrame(dataset2_sorted)
    # test_data.to_csv(test_data_path, index=False)
    # print('train_data saved at: ', test_data_path)

    # dataset3 = pd.DataFrame()
    # for i in range (dataset2_sorted.shape[0] - 3):
    #     if dataset2_sorted[i, 1] == dataset2_sorted[i+3, 1]:
    #         dataset2_sorted[i, 49] = dataset2_sorted[i+3, 49]
    #         dataset2_sorted[i, 50] = dataset2_sorted[i+3, 50]
    #         dataset3 = pd.concat([dataset3, pd.DataFrame([dataset2_sorted[i]])], ignore_index=True)

    test_data = pd.DataFrame(dataset2_sorted)
    test_data.to_csv(test_data_path, index=False)
    print('train_data saved at: ', test_data_path)
    
    # training_set = pd.DataFrame()
    # validation_set = pd.DataFrame()
    
    # game_versions = dataset3[0].unique()

    # for fifa_version in game_versions:
    #     current_version_data = dataset3[dataset3[0] == fifa_version]
    #     train_data, validation_data = train_test_split(current_version_data, test_size=0.2, random_state=42)

    #     training_set = pd.concat([training_set, train_data], ignore_index=True)
    #     validation_set = pd.concat([validation_set, validation_data], ignore_index=True)

    # print("Train set shape:", training_set.shape)
    # print("Validation set shape:", validation_set.shape)

    # X_temp = np.zeros((8101, 14))
    # for i in range (14):
    #     X_temp[:, i] = dataset1[:, lst_numerical[i]].astype(float)

    # X_result = scale(X_temp)               # scale the numerical data

    # for i in range(14):
    #     dataset2[:, lst_numerical[i]] = X_result[:, i]

    # feature_data = dataset2[:, :48]
    # label_data1 = dataset2[:, 48]
    # label_data2 = dataset2[:, 49] 

    # games = ["fifa17", "fifa18", "fifa19", "fifa20", "fifa21", "fifa22"]

    # for game, (start, end) in zip(games, row_ranges):
    #     game_data = dataset3.iloc[start:end, :]

    #     train_set, validation_set = train_test_split(game_data, test_size=0.2, random_state=42)

    #     training_data = pd.concat([training_data, train_set], ignore_index=True)
    #     validation_data = pd.concat([validation_data, validation_set], ignore_index=True)

    # print("Training set shape:", training_data.shape)
    # print("Validation set shape:", validation_data.shape)

    # x_train, x_validation, y_train, y_validation = train_test_split(feature_data, label_data1, test_size = 0.2)     # split the data into training set and validation set
                                                                   
    # dataset_train = np.zeros((6480, 20))
    # dataset_validation = np.zeros((1621, 20))

    # dataset_train[:, :19] = x_train[:, :19]
    # dataset_train[:, 19] = y_train
    # dataset_validation[:, :19] = x_validation[:, :19]
    # dataset_validation[:, 19] = y_validation

    # final_train_data = pd.DataFrame(training_set)
    # final_train_data.to_csv(train_data_path, index=False)
    # print('train_data saved at: ', train_data_path)           # save the training set file 

    # final_validation_data = pd.DataFrame(validation_set)
    # final_validation_data.to_csv(validation_data_path, index=False)
    # print('validation_data saved at: ', validation_data_path)      # save the validation set file 



    