import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler


def process_event_data(event_data, full_data):
    """
    Process event data to create a DataFrame with event types for each frame.
    
    Parameters:
    -----------
    event_data : pd.DataFrame
        DataFrame containing event information with Start Frame, End Frame, and Type columns
    full_data : pd.DataFrame
        Complete dataset with all frames
        
    Returns:
    --------
    pd.DataFrame
        Processed event data with frame-by-frame event types
    """
    ball_out_idx = full_data.index[
        (full_data["Ball-x"] < 0) | (full_data["Ball-x"] > 1) |
        (full_data["Ball-y"] < 0) | (full_data["Ball-y"] > 1)
    ]

    ball_out_df = pd.DataFrame({
        "Type": "BALL OUT"
    }, index=ball_out_idx)

    event_data = event_data[~(event_data["Type"] == "BALL OUT")]
    start_frames = event_data["Start Frame"].iloc[1:].to_numpy()
    end_frames = event_data["End Frame"].iloc[1:].to_numpy()
    event_types = event_data["Type"].iloc[1:].to_numpy()

    # Validation
    assert start_frames.shape == end_frames.shape == event_types.shape

    # Ensure end frames don't exceed data bounds
    end = full_data.index[-1]
    end_frames = np.minimum(end_frames, end)

    # Create frame ranges and unique indices
    frame_ranges = [np.arange(min(i, j), max(i, j) + 1)
                    for i, j in zip(start_frames, end_frames)]
    unique_indices = np.unique(np.concatenate(frame_ranges))

    # Create and populate event DataFrame
    event_df = pd.DataFrame(index=unique_indices, columns=["Type"])
    for s, e, e_t in zip(start_frames, end_frames, event_types):
        event_df.loc[s:e, "Type"] = e_t
        
    combined_event_df = ball_out_df.combine_first(event_df)
    
    return combined_event_df.dropna()


def plot_confusion_matrix(y_train, y_train_pred, y_test, y_pred, labels, split, model_name=""):
    """
    Plot confusion matrix and print accuracy scores.
    
    Parameters:
    -----------
    y_train : array-like
        True training labels
    y_train_pred : array-like
        Predicted training labels
    y_test : array-like
        True test labels
    y_pred : array-like
        Predicted test labels
    labels : list
        List of label names
    split : int
        Split number for cross-validation
    model_name : str, optional
        Name of the model for plot title
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    cm_counts = confusion_matrix(y_test, y_pred, labels=labels)
    cm_normalized = confusion_matrix(y_test, y_pred, labels=labels, normalize="true")

    # Create annotations with both counts and normalized values
    annot = [[f"{count} | {norm:.2f}"
              for count, norm in zip(row_counts, row_norm)]
             for row_counts, row_norm in zip(cm_counts, cm_normalized)]

    sns.heatmap(cm_normalized,
                annot=annot,
                fmt="",
                cmap="viridis",
                xticklabels=labels,
                yticklabels=labels,
                cbar_kws={"label": "Normalized Frequency"})

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{model_name.title()} Confusion Matrix Split #{(split + 1)}")
    plt.tight_layout()
    plt.show()

    # Print accuracy scores
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"{model_name} training accuracy: {train_accuracy * 100:0.2f}%")
    print(f"{model_name} testing accuracy: {test_accuracy * 100:0.2f}%\n")
    print(f"{model_name} testing precision: {precision * 100:0.2f}%")
    print(f"{model_name} testing recall: {recall * 100:0.2f}%")
    print(f"{model_name} testing f1: {f1 * 100:0.2f}%\n\n")

    return fig

def plot_accuracy_history(history: dict[str, list], title: str = ""):
    """
    Plot learning curves from training history.
    
    Parameters:
    -----------
    history : dict
        Dictionary containing metrics history
    title : str, optional
        Title for the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    for metric, results in history.items():
        num_epochs = int(len(results))
        ax.plot(list(range(1, num_epochs + 1)), results, marker="o", label=metric)

    ax.set_xlabel("Number of epochs")
    ax.set_ylabel("Training and validation accuracy")
    ax.set_title(f"Learning curve for {title}")

    ax.grid(True)
    ax.legend()

    plt.show()


def calculate_player_metrics(dataset: pd.DataFrame, calculate_velocity: bool = True,
                           calculate_acceleration: bool = True, calculate_direction: bool = True,
                           normalize: bool = True) -> pd.DataFrame:
    """
    Calculates metrics (velocity, acceleration, direction) for players and ball in a given dataset.
    
    Parameters:
        dataset (pandas.DataFrame): The input dataset containing player and ball positions over time.
        calculate_velocity (bool): Whether to calculate velocity.
        calculate_acceleration (bool): Whether to calculate acceleration.
        calculate_direction (bool): Whether to calculate direction.
        normalize (bool): Whether to normalize the calculated values.
    
    Returns:
        pandas.DataFrame: The original dataset with additional columns for selected metrics.
    """
    temp_data = dataset.copy()
    start_idx = temp_data.columns.get_loc("Time [s]")
    player_columns = temp_data.columns[start_idx + 1:]
    
    # Store all values for global scaling
    all_velocities = []
    all_accelerations = []
    
    # Dictionaries to store calculated metrics by entity
    entity_metrics = {
        'velocity': {},
        'acceleration': {},
        'direction_sin': {},
        'direction_cos': {}
    }
    
    # Calculate all values first
    for i in range(0, len(player_columns) - 1, 2):
        ply_x, ply_y = player_columns[i], player_columns[i + 1]
        
        # Identify entity
        if "ball" in str(ply_x).lower():
            entity = "Ball"
        else:
            players_num = ply_x[11]
            if len(ply_x) == 15:
                players_num = ply_x[11:13]
            entity = f"P_{players_num}"
        
        # Calculate positional differences
        x_diff = temp_data[ply_x].diff()
        y_diff = temp_data[ply_y].diff()
        time_diff = temp_data["Frame"].diff()
        
        # Calculate distance and velocity
        distance = np.sqrt(x_diff ** 2 + y_diff ** 2)
        velocity = distance / time_diff
        
        # Calculate velocity components for direction
        vel_x = x_diff / time_diff
        vel_y = y_diff / time_diff
        
        # Store values based on what metrics are needed
        if calculate_velocity:
            entity_metrics['velocity'][entity] = velocity
            all_velocities.extend(velocity.fillna(0).tolist())
        
        # Calculate acceleration if needed
        if calculate_acceleration:
            acceleration = velocity.diff() / time_diff
            entity_metrics['acceleration'][entity] = acceleration
            all_accelerations.extend(acceleration.fillna(0).tolist())
        
        # Calculate direction components if needed
        if calculate_direction:
            direction_rad = np.arctan2(vel_y, vel_x)
            entity_metrics['direction_sin'][entity] = np.sin(direction_rad)
            entity_metrics['direction_cos'][entity] = np.cos(direction_rad)
    
    # Apply global scaling if requested
    if normalize:
        if calculate_velocity and all_velocities:
            scaler_vel = MinMaxScaler(feature_range=(-1, 1))
            scaler_vel.fit(np.array(all_velocities).reshape(-1, 1))
            
            for entity in entity_metrics['velocity']:
                entity_metrics['velocity'][entity] = scaler_vel.transform(
                    entity_metrics['velocity'][entity].values.reshape(-1, 1)
                ).flatten()
                
        if calculate_acceleration and all_accelerations:
            scaler_acc = MinMaxScaler(feature_range=(-1, 1))
            scaler_acc.fit(np.array(all_accelerations).reshape(-1, 1))
            
            for entity in entity_metrics['acceleration']:
                entity_metrics['acceleration'][entity] = scaler_acc.transform(
                    entity_metrics['acceleration'][entity].values.reshape(-1, 1)
                ).flatten()
    
    # Add calculated metrics to dataframe
    entities = set()
    for metric_type in entity_metrics:
        for entity in entity_metrics[metric_type]:
            entities.add(entity)
    
    for entity in entities:
        if calculate_velocity and entity in entity_metrics['velocity']:
            temp_data[f"{entity}_velocity"] = entity_metrics['velocity'][entity]
        
        if calculate_acceleration and entity in entity_metrics['acceleration']:
            temp_data[f"{entity}_acceleration"] = entity_metrics['acceleration'][entity]
        
        if calculate_direction:
            if entity in entity_metrics['direction_sin']:
                temp_data[f"{entity}_direction_sin"] = entity_metrics['direction_sin'][entity]
            if entity in entity_metrics['direction_cos']:
                temp_data[f"{entity}_direction_cos"] = entity_metrics['direction_cos'][entity]
    
    return temp_data


# Wrapper functions for backward compatibility
def calculate_velocity_acceleration(dataset: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
    """Wrapper for backward compatibility"""
    return calculate_player_metrics(dataset, True, True, False, normalize)

def calculate_velocity_direction(dataset: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
    """Wrapper for backward compatibility"""
    return calculate_player_metrics(dataset, True, False, True, normalize)

def calculate_velocity_acceleration_direction(dataset: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
    """Wrapper for backward compatibility"""
    return calculate_player_metrics(dataset, True, True, True, normalize)


def add_shot_prediction_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Adds features to help identify shots based on ball trajectory and goal proximity.
    
    Parameters:
        dataset (pandas.DataFrame): Dataset with ball position and direction
        
    Returns:
        pandas.DataFrame: Dataset with added shot prediction features
    """
    temp_data = dataset.copy()
    
    # Get ball position
    ball_x = temp_data["Ball-x"]
    ball_y = temp_data["Ball-y"]
    
    # Define goal positions (assuming normalized coordinates 0-1)
    left_goal_x, left_goal_y = 0.0, 0.5
    right_goal_x, right_goal_y = 1.0, 0.5
    goal_half_width = 0.04  # Half the goal width in normalized coordinates
    
    # Calculate distances to each goal center
    dist_to_left_goal = np.sqrt((ball_x - left_goal_x)**2 + (ball_y - left_goal_y)**2) * -1
    dist_to_right_goal = np.sqrt((ball_x - right_goal_x)**2 + (ball_y - right_goal_y)**2)
    
    # Add distance to nearest goal
    nearest_goal_dist = np.minimum(np.abs(dist_to_left_goal), dist_to_right_goal)
    temp_data["Ball_nearest_goal_distance"] = nearest_goal_dist
    
    # Determine which goal is closer (per row)
    is_left_goal_closer = dist_to_left_goal < dist_to_right_goal
    
    # Make sure we have direction features
    if "Ball_direction_sin" not in temp_data.columns or "Ball_direction_cos" not in temp_data.columns:
        raise ValueError("Ball direction features are required (Ball_direction_sin and Ball_direction_cos)")
    
    # Get ball direction
    sin_dir = temp_data["Ball_direction_sin"]
    cos_dir = temp_data["Ball_direction_cos"]
    
    # Calculate unit vector of ball direction
    dir_magnitude = np.sqrt(sin_dir**2 + cos_dir**2)
    dir_magnitude = np.where(dir_magnitude < 1e-10, 1e-10, dir_magnitude)  # Avoid division by zero
    
    unit_x = cos_dir / dir_magnitude
    unit_y = sin_dir / dir_magnitude
    
    # Calculate base probabilities
    max_distance = np.sqrt(2)  # Maximum possible distance on a unit pitch
    
    # Calculate for left goal
    t_left = np.where(np.abs(unit_x) > 1e-10, (left_goal_x - ball_x) / unit_x, np.inf)
    intersect_y_left = ball_y + t_left * unit_y
    left_goal_intersect = (t_left > 0) & (intersect_y_left >= left_goal_y - goal_half_width) & (intersect_y_left <= left_goal_y + goal_half_width)
    left_goal_probability = np.where(
        left_goal_intersect,
        np.exp(-3 * np.abs(t_left) / max_distance),
        0
    )
    in_left_penalty_area = (ball_x < 0.18) & (ball_y > 0.3) & (ball_y < 0.7)
    left_goal_shot_prob = np.where(in_left_penalty_area, np.minimum(left_goal_probability * 1.5, 1), left_goal_probability)
    
    # Calculate for right goal
    t_right = np.where(np.abs(unit_x) > 1e-10, (right_goal_x - ball_x) / unit_x, np.inf)
    intersect_y_right = ball_y + t_right * unit_y
    right_goal_intersect = (t_right > 0) & (intersect_y_right >= right_goal_y - goal_half_width) & (intersect_y_right <= right_goal_y + goal_half_width)
    right_goal_probability = np.where(
        right_goal_intersect,
        np.exp(-3 * np.abs(t_right) / max_distance),
        0
    )
    in_right_penalty_area = (ball_x > 0.82) & (ball_y > 0.3) & (ball_y < 0.7)
    right_goal_shot_prob = np.where(in_right_penalty_area, np.minimum(right_goal_probability * 1.5, 1), right_goal_probability)
    
    # Choose probability based on which goal is closer
    temp_data["Ball_shot_probability"] = np.where(is_left_goal_closer, left_goal_shot_prob, right_goal_shot_prob)
    
    return temp_data


def get_frame_data(dataset: pd.DataFrame, columns: list[int], frame: int = 1000000, frame_interval: int = 1000000,
                   feature="acceleration") -> pd.DataFrame:
    """
    Extracts frame data for ball and players within a specified interval.
    
    Args:
        dataset (pd.DataFrame): Input DataFrame containing tracking data
        columns (list[int]): List of column indices to process
        frame (int): Center frame number
        frame_interval (int): Number of frames to include before and after center frame
    
    Returns:
        pd.DataFrame: DataFrame containing positions, velocities, and accelerations
    """
    # Calculate frame range
    start_range = max(dataset.index[0], frame - frame_interval)
    end_range = min(dataset.index[-1], frame + frame_interval)

    # Initialize DataFrame with proper index
    temp_data = pd.DataFrame(index=dataset.loc[start_range:end_range].index)

    for col in columns:
        if "ball" in str(col).lower():
            ball_prefix = col[:4]
            temp_data["Ball-x"] = dataset.loc[start_range:end_range, f"{ball_prefix}-x"]
            temp_data["Ball-y"] = dataset.loc[start_range:end_range, f"{ball_prefix}-y"]
            temp_data["Ball_velocity"] = dataset.loc[start_range:end_range, "Ball_velocity"]
            temp_data[f"Ball_{feature}"] = dataset.loc[start_range:end_range, f"Ball_{feature}"]
        else:
            player_num = col[11]
            if len(str(col)) == 15:
                player_num = col[11:13]

            player_num = int(player_num)

            prefix = "Home" if player_num < 15 else "Away"

            # Add player position data
            temp_data[f"{prefix}-Player{player_num}-x"] = dataset.loc[start_range:end_range, f"{prefix}-Player{player_num}-x"]
            temp_data[f"{prefix}-Player{player_num}-y"] = dataset.loc[start_range:end_range, f"{prefix}-Player{player_num}-y"]

            # Add player movement data
            temp_data[f"P_{player_num}_velocity"] = dataset.loc[start_range:end_range, f"P_{player_num}_velocity"]
            temp_data[f"P_{player_num}_{feature}"] = dataset.loc[start_range:end_range, f"P_{player_num}_{feature}"]

    return temp_data


def calculate_player_ball_distances(game_data, player_data, ball_data):
    """
    Calculate distances between players and the ball for each timestamp.
    
    Parameters:
    game_data (pd.DataFrame): DataFrame containing game timestamps
    player_data (pd.DataFrame): DataFrame containing player positions
    ball_data (pd.DataFrame): DataFrame containing ball positions
    
    Returns:
    pd.DataFrame: DataFrame with calculated distances for each player
    """
    index = game_data.index
    result = pd.DataFrame(index=index)
    result["Time [s]"] = game_data.loc[index, "Time [s]"]

    # Calculate distance for each player
    for i in range(0, player_data.shape[1] - 1, 2):
        player_x = player_data.loc[index, player_data.columns[i]]
        player_y = player_data.loc[index, player_data.columns[i + 1]]

        # Euclidean distance calculation
        euclidean_x = np.square(player_x - ball_data.loc[index, "Ball-x"])
        euclidean_y = np.square(player_y - ball_data.loc[index, "Ball-y"])
        distance = np.sqrt(euclidean_x + euclidean_y)

        # Store result using player name without coordinate suffix
        player_name = player_data.columns[i][:-2]
        result[player_name] = distance

    return result

def expand_dataset(dataset: pd.DataFrame | pd.Series, look_back: int = 5) -> pd.DataFrame:
    """
    Expands the given dataset by creating overlapping windows of data points for a specified look-back period.

    Args:
        dataset (pd.DataFrame): The input dataset to expand.
        look_back (int): The number of past observations to include in each window. Default is 5.

    Returns:
        pd.DataFrame: The expanded dataset with overlapping windows.
    """

    if look_back < 1 or look_back >= len(dataset):
        raise ValueError(f"look_back must be between 1 and {len(dataset) - 1}.")

    data = dataset.values
    if data.ndim == 1:
        data = data[:, None]
        
    indices = np.arange(len(dataset) - look_back)[:, None] + np.arange(look_back + 1)
    
    sequences = data[indices]

    if isinstance(dataset, pd.Series):
        columns = [dataset.name] if dataset.name is not None else [0]
    else:
        columns = dataset.columns
    
    expanded_df = pd.DataFrame(
        sequences.reshape(-1, data.shape[1]), 
        columns=columns
    )
    
    return expanded_df


def track_closest_players(game_data: pd.DataFrame, closest_players: pd.DataFrame) -> pd.DataFrame:
    """
    Track positions of players closest to the ball.
    
    Parameters:
    game_data (pd.DataFrame): DataFrame containing game data
    closest_players (pd.Series/np.array): Array of player names who are closest to ball
    
    Returns:
    pd.DataFrame: DataFrame with positions of closest players and ball
    """
    assert game_data.shape[0] == closest_players.shape[0], f"The shape of game_data: {game_data.shape} is different from closest_players: {closest_players.shape}"
    
    game_data_pl = pl.from_pandas(game_data)
    closest_players_pl = pl.from_pandas(closest_players)

    data_rows = []
    player_columns = [[f"P_{i}-x", f"P_{i}-y"] for i in range(1, closest_players.shape[1] + 1)]
    player_columns = np.array(player_columns).ravel().tolist()
    
    # Create arrays of column names for x and y coordinates
    x_columns = np.array([[f"{player}-x" for player in players_list] for players_list in closest_players_pl.to_numpy()])
    y_columns = np.array([[f"{player}-y" for player in players_list] for players_list in closest_players_pl.to_numpy()])
    
    columns = np.empty((x_columns.shape[0], x_columns.shape[1] * 2), dtype=object)
    columns[:, 0::2] = x_columns
    columns[:, 1::2] = y_columns
    
    for idx in range(len(game_data_pl)):
        temp = game_data_pl.select(pl.col(columns[idx].tolist())).row(idx)
        data_rows.append(temp)

    # Create the DataFrame with the collected data
    min_dist_to_ball = pl.DataFrame(
        data_rows,
        schema=player_columns,
        orient="row"
    )
    
    min_dist_to_ball.insert_column(0, pl.Series("Time [s]", game_data_pl["Time [s]"]))
    min_dist_to_ball.insert_column(1, pl.Series("Ball-x", game_data_pl["Ball-x"]))
    min_dist_to_ball.insert_column(2, pl.Series("Ball-y", game_data_pl["Ball-y"]))

    min_dist_to_ball = min_dist_to_ball.to_pandas()
    min_dist_to_ball.index = game_data.index
    
    return min_dist_to_ball


def get_n_smallest_indices_sorted(df, n=5):
    arr = df.values
    # Get indices of n smallest elements (sorted)
    idx = np.argsort(arr, axis=1)[:, :n]
    result = []
    for row in idx:
        result.append(df.columns[row].tolist())

    columns = []
    for i in range(1, n + 1):
        columns.append(f"P-{i}")
        
    return pd.DataFrame(result, index=df.index, columns=columns)

