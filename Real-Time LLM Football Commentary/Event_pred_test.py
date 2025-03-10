import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_score, recall_score
from matplotlib import pyplot as plt
import seaborn as sns

from kloppy import metrica, wyscout

games = [2576322, 2576323, 2576324, 2576325, 2576326, 2576327, 2576328, 2576329, 2576330, 2576331, 2576333, 2576334, 2576335, 2576336, 2576337, 2576338, 2576290, 2576291, 2576292, 2576293, 2576294, 2576295, 2576296, 
         2576297, 2576298, 2576299, 2576300, 2576301, 2576302, 2576303, 2576304, 2576305, 2576306, 2576307, 2576308, 2576272, 2576273, 2576274, 2576275, 2576276, 2576277, 2576278, 2576279, 2576280, 2576282, 2576283, 
         2576284, 2576285, 2576286, 2576287, 2576288, 2576289, 2516893, 2516894, 2516895, 2516896, 2516897, 2516898, 2516899, 2516900, 2516901, 2516902, 2516903, 2516904, 2516905, 2516906, 2516907, 2516908, 2516909, 
         2516910, 2516911, 2516912, 2516873, 2516874, 2516875, 2516876, 2516877, 2516878, 2516879, 2516880, 2516881, 2516882, 2516883, 2516884, 2516885, 2516886, 2516887, 2516888, 2516889, 2516890, 2516891, 2516892,
         2516913, 2516914, 2516915, 2516916, 2516917, 2516918, 2516919, 2516920, 2516921, 2516922, 2516923, 2516924, 2516926, 2516927, 2516928, 2516929, 2516930, 2516931, 2516932]


def game_loader(matches):
    result = pd.DataFrame([])
    for i in matches:
        dataset = wyscout.load_open_data(match_id=i)
        game = dataset.to_df()
        temp_game = game.copy()
        temp_game.dropna(axis=1, how="all", inplace=True)
        temp_game = temp_game[["event_type", "coordinates_x", "coordinates_y", "end_coordinates_x", "end_coordinates_y"]]
        temp_game = temp_game[~(temp_game["event_type"] == "GENERIC:generic")]
        
        cols_with_nan = temp_game.columns[temp_game.isnull().any()]
        for col in cols_with_nan:
            idxs = temp_game[temp_game[col].isnull()].index
            if col.endswith("x"):
                temp_game.loc[idxs, col] = temp_game.loc[idxs, "coordinates_x"]
            elif col.endswith("y"):
                temp_game.loc[idxs, col] = temp_game.loc[idxs, "coordinates_y"]

        result = pd.concat([result, temp_game])
        
    return result