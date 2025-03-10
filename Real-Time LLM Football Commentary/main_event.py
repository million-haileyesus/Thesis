import json
import numpy as np
import pandas as pd

from utils import process_event_data

def prepare_event_data(data, game_data):
    game_event_data = pd.read_csv(data)    
    game_event_data = game_event_data[~(game_event_data["Type"] == "FAULT RECEIVED")]
    game_event_data = game_event_data[~(game_event_data["Type"] == "SET PIECE")]
    game_event_data = game_event_data.replace("RECOVERY", "BALL LOST")    
    game_event_data = game_event_data[["Type", "Start Frame", "End Frame", "From", "To"]]
    game_event = process_event_data(game_event_data, game_data)

    return game_event

def prepare_json_event_data(data, game_data):
    with open(data, "r") as f:
        event = json.load(f)
        
    type_ = np.array([])
    start_frame = np.array([])
    end_frame = np.array([])
    
    for i in event["data"]:
        t = i["type"]["name"]
        sf = i["start"]["frame"]
        ef = i["end"]["frame"]
    
        if t != "CARRY" and t != "BALL OUT":
            type_ = np.append(type_, t)
            start_frame = np.append(start_frame, sf)
            end_frame = np.append(end_frame, ef)
    
    
    game_event_data = pd.DataFrame({"Type": type_, "Start Frame": start_frame.astype(np.int64), "End Frame": end_frame.astype(np.int64)})
    game_event_data = game_event_data[~(game_event_data["Type"] == "FAULT RECEIVED")]
    game_event_data = game_event_data[~(game_event_data["Type"] == "SET PIECE")]
    
    game_event_data = game_event_data.replace("RECOVERY", "BALL LOST")
    game_event = process_event_data(game_event_data, game_data)
    
    return game_event