import copy
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import Metrica_IO as mio
import Metrica_Viz as mviz
from config import Config


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class PreProcessing:
    """
    A class for preprocessing soccer match tracking data. Handles tasks such as
    reading data files, adding ball data, fixing headers, and handling missing
    values for both home and away team data.
    """

    def __init__(self):
        """
        Initialize the PreProcessing class.
        """
        self.data_home_path = None
        self.data_away_path = None

        self._config = Config()
        self._colors = self._config.COLOR_MAP

    def _validate_input_files(self) -> None:
        """Validate that input files exist and are readable."""
        if not self.data_home_path.exists():
            raise FileNotFoundError(f"Home team data file not found: {self.data_home_path}")
        if not self.data_away_path.exists():
            raise FileNotFoundError(f"Away team data file not found: {self.data_away_path}")

    def load_and_process_data(
            self, data_home: str | Path, data_away: str | Path,
            add_ball_data: bool = True, add_headers: bool = True,
            half_period: str | int = "both", remove_ball_nan: bool = True
    ) -> pd.DataFrame:
        """
        Load and process match data from both teams.

        Args:
            data_home: Path to home team data file
            data_away: Path to away team data file
            add_ball_data: Whether to add ball tracking data
            add_headers: Whether to add customized column headers
            half_period: Which half of the match to process (1, 2, or both)
            remove_ball_nan: Whether to remove rows with missing ball data

        Returns:
            Processed DataFrame combining both teams' data
        """
        self.data_home_path = Path(data_home)
        self.data_away_path = Path(data_away)

        self._validate_input_files()

        self._validate_parameters(half_period)

        original_df_home = self._read_file(self.data_home_path)
        original_df_away = self._read_file(self.data_away_path)

        df_home = self._remove_headers(original_df_home)
        df_away = self._remove_headers(original_df_away)

        df_subs_home = self.get_subs(original_df_home)
        df_subs_away = self.get_subs(original_df_away)

        if add_ball_data:
            df_home = self._add_ball_data(original_df_home, df_home)
            df_away = self._add_ball_data(original_df_away, df_away)

            df_subs_home = self._add_ball_data(original_df_home, df_subs_home)
            df_subs_away = self._add_ball_data(original_df_away, df_subs_away)

        if add_headers:
            df_home.columns = self._generate_headers(dataset=df_home, team="Home", add_ball_data=add_ball_data)
            df_away.columns = self._generate_headers(dataset=df_away, team="Away", add_ball_data=add_ball_data)

            df_subs_home.columns = self._generate_headers(dataset=df_subs_home, team="Home", add_ball_data=add_ball_data)
            df_subs_away.columns = self._generate_headers(dataset=df_subs_away, team="Away", add_ball_data=add_ball_data)

        df_home = self._convert_to_numeric(df_home)
        df_away = self._convert_to_numeric(df_away)

        df_subs_home = self._convert_to_numeric(df_subs_home)
        df_subs_away = self._convert_to_numeric(df_subs_away)

        df_home = self._choose_halfs(df_home, df_subs_home, half_period)
        df_away = self._choose_halfs(df_away, df_subs_away, half_period)

        if add_ball_data:
            df_home = self._filter_nan_data(df_home, remove_ball_nan)
            df_away = self._filter_nan_data(df_away, remove_ball_nan)

        home_away_data = self._merge_team_data(df_home, df_away, add_ball_data)

        return home_away_data

    def _validate_parameters(self, half_period: int) -> None:
        """Validate input parameters."""
        if half_period not in (1, 2, "both"):
            raise ValueError("half_period must be either 1, 2 or both")

    def _read_file(self, file_path: Path) -> pd.DataFrame:
        """Read a single team's data file."""
        try:
            return pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            raise DataValidationError(f"Error reading {file_path}: {str(e)}")

    def _remove_headers(self, dataset: pd.DataFrame) -> pd.DataFrame:
        temp_data = dataset.iloc[2:].reset_index(drop=True)
        temp_data.columns = dataset.iloc[1].values
        temp_data = temp_data.drop(columns=temp_data.columns[25:])
        temp_data.index = range(1, len(temp_data) + 1)

        return temp_data

    def get_subs(self, dataset: pd.DataFrame) -> pd.DataFrame:
        temp_data = dataset.iloc[2:].reset_index(drop=True)
        temp_data.columns = dataset.iloc[1].values
        columns_to_drop = list(temp_data.columns[3:25]) + list(temp_data.columns[-2:])
        temp_data = temp_data.drop(columns=columns_to_drop)
        temp_data.index = range(1, len(temp_data) + 1)

        return temp_data

    def _add_ball_data(self, original_dataset: pd.DataFrame, player_dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Addind ball tracking data to the dataset.

        Args:
            original_dataset: Input DataFrame containing all the data
            player_dataset: Input DataFrame containing raw player tracking data

        Returns:
            DataFrame with processed ball data
        """
        ball = pd.concat([pd.to_numeric(original_dataset.iloc[:, -2], errors='coerce'),
                          pd.to_numeric(original_dataset.iloc[:, -1], errors='coerce')], axis=1).iloc[2:]
        ball.index = range(1, len(ball) + 1)

        result = pd.concat([player_dataset, ball], axis=1)

        return result

    def _generate_headers(self, dataset: pd.DataFrame, team: str, add_ball_data: bool) -> list[str]:
        columns = list(dataset.columns)
        headers = columns[:3]

        e = len(columns)
        if add_ball_data:
            e -= 2

        end = e if len(columns) < 25 else 25  # This part is for add the ball
        for i in range(3, end):
            if isinstance(columns[i], str):
                headers.append(f"{team}-{columns[i]}-x")
            else:
                headers.append(f"{team}-{columns[i - 1]}-y")

        if add_ball_data:
            headers.extend(["Ball-x", "Ball-y"])

        return headers

    def _convert_to_numeric(self, dataset: pd.DataFrame) -> pd.DataFrame:
        temp_data = copy.deepcopy(dataset)

        for col in temp_data.columns:
            if col == "Period" or col == "Frame":
                temp_data[col] = pd.to_numeric(temp_data[col], errors='coerce').astype("Int64")
            else:
                temp_data[col] = pd.to_numeric(temp_data[col], errors='coerce')

        return temp_data

    def _choose_halfs(self, dataset: pd.DataFrame, subs_dataset: pd.DataFrame, half_period: str | int = 1) -> pd.DataFrame:
        temp_data = copy.deepcopy(dataset)

        if half_period == 2 or (isinstance(half_period, str) and half_period.lower() == "both"):
            self._replace_subs(temp_data, subs_dataset)
        else:
            temp_data = temp_data[temp_data["Period" == half_period]]

        return temp_data

    def _replace_subs(self, dataset: pd.DataFrame, subs_dataset: pd.DataFrame) -> pd.DataFrame:
        for i in range(3, dataset.shape[1]):
            idxs = dataset[dataset.iloc[:, i].isna()].index
            if idxs.shape[0] > 0 and not dataset.columns[i].startswith("Ball"):
                x = idxs[0]
                for j in range(3, subs_dataset.shape[1]):
                    d_col = dataset.columns[i]
                    s_col = subs_dataset.columns[j]
                    if dataset.loc[x - 1, d_col] == subs_dataset.loc[x - 1, s_col]:
                        dataset.loc[idxs, d_col] = subs_dataset.loc[idxs, s_col]
                        break

        return dataset

    def _filter_nan_data(self, dataset: pd.DataFrame, remove_ball_nan: bool) -> pd.DataFrame:
        """
        Removing or filling NaN values

        Args:
            dataset: Input DataFrame with missing values
            half_period: Which half of the match to process (1 or 2)
            remove_ball_nan: Whether to remove rows with missing ball data

        Returns:
            DataFrame with NaN values filled
        """
        if remove_ball_nan:
            # Dropping the rows where the ball's x-y coordinates are NaN
            period_data = dataset.dropna(subset=["Ball-x", "Ball-y"], how="any")
        else:
            period_data = self._fill_missing_with_interpolation_and_fill(dataset)

        return period_data

    def _fill_missing_with_interpolation_and_fill(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Fill NaN values in numeric columns using linear interpolation,
        followed by forward and backward filling for remaining NaNs.

        Args:
            dataset (pd.DataFrame): Input DataFrame with potential NaN values.

        Returns:
            pd.DataFrame: DataFrame with NaN values filled.
        """
        # Create a copy of the input DataFrame to avoid modifying the original
        df_filled = dataset.copy()

        # Iterate through each column
        for col in df_filled.columns:
            if pd.api.types.is_numeric_dtype(df_filled[col]):
                df_filled[col] = df_filled[col].interpolate(method="linear")
            df_filled[col] = df_filled[col].ffill().bfill()

        return df_filled

    def _merge_team_data(self, data_home: pd.DataFrame, data_away: pd.DataFrame, add_ball_data: bool) -> pd.DataFrame:
        """Merge home and away team data into a single DataFrame."""
        if add_ball_data:
            return pd.concat([
                data_home.iloc[:, :-2],  # Exclude ball columns from home
                data_away.iloc[:, 3:]  # Exclude Period, Frame, Time from away
            ], axis=1)
        else:
            return pd.concat([
                data_home.iloc[:, :],
                data_away.iloc[:, 3:]  # Exclude Period, Frame, Time from away
            ], axis=1)

    def expand_dataset(self, dataset: pd.DataFrame | pd.Series, look_back: int = 5) -> pd.DataFrame:
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

    def vel_acc(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the velocity and acceleration of players and ball in a given dataset.

        Parameters:
        dataset (pandas.DataFrame): The input dataset containing player and ball positions over time.

        Returns:
        pandas.DataFrame: The original dataset with additional columns for velocity and acceleration.
        """
        temp_data = copy.deepcopy(dataset)
        star_idx = temp_data.columns.get_loc("Time [s]")
        player_columns = temp_data.columns[star_idx + 1:]

        for i in range(0, player_columns.shape[0] - 1, 2):
            # Calculate Euclidean distance between consecutive points
            ply_x, ply_y = player_columns[i], player_columns[i + 1]

            x_diff = temp_data[ply_x].diff()
            y_diff = temp_data[ply_y].diff()

            # Calculate time difference between frames
            time_diff = temp_data["Time [s]"].diff()

            # Distance calculation
            distance = np.sqrt(x_diff ** 2 + y_diff ** 2)

            # Velocity calculation
            velocity = distance / time_diff

            # Acceleration calculation
            acceleration = velocity.diff() / time_diff

            if "Ball" in ply_x:
                temp_data[f"Ball_velocity"] = velocity
                temp_data[f"Ball_acceleration"] = acceleration
            else:
                players_num = ply_x[11]
                if len(ply_x) == 15:
                    players_num = ply_x[11:13]

                temp_data[f"P_{players_num}_velocity"] = velocity
                temp_data[f"P_{players_num}_acceleration"] = acceleration

        return temp_data

    def vel_dir(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the velocity and direction of players and ball in a given dataset.
    
        Parameters:
        dataset (pandas.DataFrame): The input dataset containing player and ball positions over time.

        Returns:
        pandas.DataFrame: The original dataset with additional columns for velocity and direction.
        """
        temp_data = copy.deepcopy(dataset)
        start_idx = temp_data.columns.get_loc("Time [s]")
        player_columns = temp_data.columns[start_idx + 1:]
    
        for i in range(0, player_columns.shape[0] - 1, 2):
            # Calculate Euclidean distance between consecutive points
            ply_x, ply_y = player_columns[i], player_columns[i + 1]
            
            x_diff = temp_data[ply_x].diff()
            y_diff = temp_data[ply_y].diff()

            distance = np.sqrt(x_diff**2 + y_diff**2)
            
            # Calculate time difference between frames
            time_diff = temp_data["Time [s]"].diff()
            
            # Calculate speed (distance / time)
            # Note: First row will be NaN as we can't calculate speed for a single point
            velocity = distance / time_diff
            direction = np.arctan2(x_diff, y_diff)
    
            if "ball" in str(ply_x).lower():
                temp_data[f"Ball_velocity"] = velocity
                temp_data[f"Ball_direction"] = direction
            else:
                players_num = ply_x[11]
                if len(ply_x) == 15:
                    players_num = ply_x[11:13]
    
                temp_data[f"P_{players_num}_velocity"] = velocity
                temp_data[f"P_{players_num}_direction"] = direction
    
        return temp_data

    def get_frames(self, dataset: pd.DataFrame, columns: list[int], frame: int = 1000000, frame_interval: int = 1000000, feature="acceleration") -> pd.DataFrame:
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

                if player_num < 15:
                    prefix = "Home"
                else:
                    prefix = "Away"

                    # Add player position data
                temp_data[f"{prefix}-Player{player_num}-x"] = dataset.loc[start_range:end_range, f"{prefix}-Player{player_num}-x"]
                temp_data[f"{prefix}-Player{player_num}-y"] = dataset.loc[start_range:end_range, f"{prefix}-Player{player_num}-y"]

                # Add player movement data
                temp_data[f"P_{player_num}_velocity"] = dataset.loc[start_range:end_range, f"P_{player_num}_velocity"]
                temp_data[f"P_{player_num}_{feature}"] = dataset.loc[start_range:end_range, f"P_{player_num}_{feature}"]

        return temp_data

    def player_visualization(
            self, dataset: pd.DataFrame, players: list[int] = [11], 
            sides: list[str] = ["Home"], marker_size: int = 7,
            plot_ball: bool = True, use_annotation: bool = False
    ) -> None:
        fig, ax = mviz.plot_pitch()
        ball_is_not_there = plot_ball

        title = "Player, and Side at each intervals"
        if use_annotation:
            title = "Player, and Frame at each intervals"

        for i, (player, side) in enumerate(zip(players, sides)):
            x = f"{side}-Player{player}-x"
            y = f"{side}-Player{player}-y"

            # Prepare player positions and their coordinates on the pitch
            data_x = pd.to_numeric(dataset[x], errors='coerce')
            data_y = pd.to_numeric(dataset[y], errors='coerce')

            # Create DataFrames with a single column for x and y coordinate of a player
            positions_x = pd.DataFrame(data_x, columns=[x])
            positions_y = pd.DataFrame(data_y, columns=[y])

            # Changing the dataset to metric or pitch coordinates
            positions_x = mio.to_metric_coordinates(positions_x)
            positions_y = mio.to_metric_coordinates(positions_y)

            color = self._colors[i % len(self._colors)]

            # Plot player movement with arrows
            for j in range(len(positions_x) - 1):
                dx = positions_x[x].iloc[j + 1] - positions_x[x].iloc[j]
                dy = positions_y[y].iloc[j + 1] - positions_y[y].iloc[j]

                if abs(dx) > 0.1 or abs(dy) > 0.1:  # Only draw arrows for significant movements
                    ax.arrow(positions_x[x].iloc[j], positions_y[y].iloc[j], dx, dy, head_width=1.2, head_length=1.2,
                             fc=color, ec=color, linestyle="dotted", length_includes_head=True)

            # Plot the player path with dots and lines
            if use_annotation:
                for i in range(len(positions_x)):
                    ax.plot(positions_x[x].iloc[i], positions_y[y].iloc[i], marker=".", linestyle="-.",
                            markersize=marker_size, color=color, zorder=3,
                            label=f"P{use_annotation[i]} - {positions_x.index[i]}")
                    ax.text(positions_x[x].iloc[i], positions_y[y].iloc[i], positions_x.index[i], fontsize=12,
                            ha='right')
            else:
                ax.plot(positions_x[x], positions_y[y], marker=".", linestyle="-.", markersize=marker_size, color=color,
                        zorder=3, label=f"Player {player} - {side}")
            ax.plot(positions_x[x].iloc[0], positions_y[y].iloc[0], marker=".", linestyle="-.",
                    markersize=int(marker_size * 1.75), color="green", zorder=3)
            ax.plot(positions_x[x].iloc[-1], positions_y[y].iloc[-1], marker=".", linestyle="-.",
                    markersize=int(marker_size * 1.75), color="red", zorder=3)

        # ax.annotate("Start Point", (positions_x[x].iloc[0], positions_y[y].iloc[0]), xytext=(5, 5), textcoords="offset points", fontsize=8, color=color)
        # ax.annotate("End Point", (positions_x[x].iloc[-1], positions_y[y].iloc[-1]), xytext=(5, 5), textcoords="offset points", fontsize=8, color=color)

        if ball_is_not_there:
            ba_x = pd.to_numeric(dataset["Ball-x"], errors='coerce')
            ba_y = pd.to_numeric(dataset["Ball-y"], errors='coerce')

            ba_x = mio.to_metric_coordinates(pd.DataFrame(ba_x, columns=["Ball-x"]))
            ba_y = mio.to_metric_coordinates(pd.DataFrame(ba_y, columns=["Ball-y"]))

            ax.plot(ba_x["Ball-x"], ba_y["Ball-y"], marker=".", linestyle="-.", markersize=int(marker_size * 1.5),
                    color="black", zorder=2, label="Ball")
            ax.plot(ba_x["Ball-x"].iloc[0], ba_y["Ball-y"].iloc[0], marker=".", linestyle="-.",
                    markersize=int(marker_size * 1.75), color="green", zorder=2)
            ax.plot(ba_x["Ball-x"].iloc[-1], ba_y["Ball-y"].iloc[-1], marker=".", linestyle="-.",
                    markersize=int(marker_size * 1.75), color="red", zorder=2)

            # Ball movement arrows
            for j in range(len(ba_x) - 1):
                dx = ba_x["Ball-x"].iloc[j + 1] - ba_x["Ball-x"].iloc[j]
                dy = ba_y["Ball-y"].iloc[j + 1] - ba_y["Ball-y"].iloc[j]
                if abs(dx) > 0.1 or abs(dy) > 0.1:  # Only draw arrows for significant movements
                    ax.arrow(ba_x["Ball-x"].iloc[j], ba_y["Ball-y"].iloc[j], dx, dy, head_width=1.2, head_length=1.2,
                             fc='black', ec='black', linestyle="dotted", length_includes_head=True)

        # ax.annotate("Start Point", (ba_x["ball-x"].iloc[0], ba_y["ball-y"].iloc[0]), xytext=(5, 5), textcoords="offset points", fontsize=8, color="black")
        # ax.annotate("End Point", (ba_x["ball-x"].iloc[-1], ba_y["ball-y"].iloc[-1]), xytext=(5, 5), textcoords="offset points", fontsize=8, color="black")

        start_seconds = dataset["Time [s]"].iloc[0]
        end_seconds = dataset["Time [s]"].iloc[-1]
        pl = [i for i in players]
        ax.set_title(
            f"Tracking for Player {pl} in the First Half from {start_seconds} seconds to {end_seconds} seconds")
        ax.legend(title=title, loc="upper left", fontsize=8)

        plt.tight_layout()
        plt.show()
