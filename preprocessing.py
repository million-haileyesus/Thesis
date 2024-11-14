import copy

import pandas as pd
from matplotlib import pyplot as plt

import Metrica_IO as mio
import Metrica_Viz as mviz


class PreProcessing:
    color_map = [
        "#FFA500",  # Orange
        "#0000FF",  # Blue
        "#800080",  # Purple
        "#00FFFF",  # Cyan
        "#FF00FF",  # Magenta
        "#008000",  # Green
        "#FFC0CB",  # Pink
        "#8B4513",  # Saddle Brown
        "#FFD700",  # Gold
        "#00FF00",  # Lime
        "#FF0000",  # Red
    ]
    BALL_POSSESSION_SPEED = 0.05
    MPS_TO_MPH = 2.23694
    PLAYER_NAMES = ["P_1", "P_2", "P_3", "P_4", "P_5", "P_6", "P_7", "P_8", "P_9", "P_10", "P_11", "P_25",
                    "P_15", "P_16", "P_17", "P_18", "P_19", "P_20", "P_21", "P_22", "P_23", "P_24"]

    def __init__(self, data_home: str, data_away: str):
        """
        A class for preprocessing soccer match tracking data.

        Handles tasks such as reading data files, adding ball data, fixing headers,
        and handling missing values for both home and away team data.
        """
        self.data_home_path = data_home
        self.data_away_path = data_away

    def load_and_process_data(
            self, add_ball: bool = True, add_headers: bool = True,
            half_period: int = 1, remove_ball_nan: bool = True
    ) -> pd.DataFrame:
        """
        Load and process match data from both teams.

        Args:
            add_ball: Whether to add ball tracking data
            add_headers: Whether to add customized column headers
            half_period: Which half of the match to process (1 or 2)
            remove_ball_na: Whether to remove rows with missing ball data

        Returns:
            Processed DataFrame combining both teams' data
        """
        if half_period not in (1, 2):
            raise ValueError("half_period must be either 1 or 2")
            
        original_df_home = pd.read_csv(self.data_home_path, low_memory=False)
        original_df_away = pd.read_csv(self.data_away_path, low_memory=False)

        df_home = self._remove_headers(original_df_home)
        df_away = self._remove_headers(original_df_away)

        if add_ball:
            ball = pd.concat([pd.to_numeric(original_df_home["Unnamed: 31"], errors='coerce'), 
                              pd.to_numeric(original_df_home["Unnamed: 32"], errors='coerce')], axis=1).iloc[2:]
            ball.index = range(1, len(ball) + 1)
            df_home = self._add_ball_data(df_home, ball)
            df_away = self._add_ball_data(df_away, ball)

        if add_headers:
            df_home.columns = self._generate_headers(team="Home", add_ball=add_ball, start=2, end=12)
            df_away.columns = self._generate_headers(team="Away", add_ball=add_ball, start=12, end=23)

        df_home = self._convert_to_numeric(df_home)
        df_away = self._convert_to_numeric(df_away)

        df_home = df_home[df_home["Period"] == half_period]
        df_away = df_away[df_away["Period"] == half_period]

        if add_ball:
            df_home = self._filter_nan_data(df_home, remove_ball_nan)
            df_away = self._filter_nan_data(df_away, remove_ball_nan)

        home_away_data = self._merge_team_data(df_home, df_away, add_ball)

        return home_away_data

    def _remove_headers(self, dataset: pd.DataFrame) -> pd.DataFrame:
        temp_data = copy.deepcopy(dataset)
        temp_data = temp_data.iloc[2:, :-2].reset_index(drop=True)
        columns_to_drop = temp_data.columns[(temp_data.iloc[0].isna())]
        temp_data = temp_data.drop(columns=columns_to_drop)
        temp_data.index = range(1, len(temp_data) + 1)
        
        return temp_data

    def _add_ball_data(self, dataset: pd.DataFrame, ball_data: pd.DataFrame) -> pd.DataFrame:
        """
        Addind ball tracking data to the dataset.

        Args:
            dataset: Input DataFrame containing raw player tracking data
            ball_data: Input DataFrame containing raw ball tracking data

        Returns:
            DataFrame with processed ball data
        """
        result = pd.concat([dataset, ball_data], axis=1)

        return result

    def _generate_headers(self, team: str, add_ball: bool, start: int, end: int) -> list[str]:
        headers = []

        if team == "Home":
            headers.extend([f"{team}-P_1-x", f"{team}-P_1-y"])

        for i in range(start, end):
            player_index = i - start + (1 if team == "Home" else 11)

            headers.append(f"{team}-{self.PLAYER_NAMES[player_index]}-x")
            headers.append(f"{team}-{self.PLAYER_NAMES[player_index]}-y")

        headers = ["Period", "Frame", "Time[s]"] + headers
        if add_ball:
            headers.extend(["ball-x", "ball-y"])

        return headers

    def _convert_to_numeric(self, dataset: pd.DataFrame) -> pd.DataFrame:
        temp_data = copy.deepcopy(dataset)

        for col in temp_data.columns:
            if col == "Period" or col == "Frame":
                temp_data[col] = pd.to_numeric(temp_data[col], errors='coerce').astype("Int64")
            else:
                temp_data[col] = pd.to_numeric(temp_data[col], errors='coerce')

        return temp_data

    def _filter_nan_data(self, dataset: pd.DataFrame, remove_ball_na: bool) -> pd.DataFrame:
        """
        Removing or filling NaN values

        Args:
            dataset: Input DataFrame with missing values
            half_period: Which half of the match to process (1 or 2)
            remove_ball_na: Whether to remove rows with missing ball data

        Returns:
            DataFrame with NaN values filled
        """
        if remove_ball_na:
            # Dropping the rows where the ball's x-y coordinates are NaN
            period_data = dataset.dropna(subset=["ball-x", "ball-y"], how="any")
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

    def _merge_team_data(self, df_home: pd.DataFrame, df_away: pd.DataFrame, add_ball: bool) -> pd.DataFrame:
        """Merge home and away team data into a single DataFrame."""
        if add_ball:
            return pd.concat([
                df_home.iloc[:, :-2],  # Exclude ball columns from home
                df_away.iloc[:, 3:]  # Exclude Period, Frame, Time from away
            ], axis=1)
        else:
            return pd.concat([
                df_home.iloc[:, :],
                df_away.iloc[:, 3:]  # Exclude Period, Frame, Time from away
            ], axis=1)

    def player_tracking(self, dataset: pd.DataFrame, players: list[int] = [11], sides: list[str] = ["Home"], marker_size: int = 7,
                        plot_ball: bool = True, use_annotation: bool = False):
        fig, ax = mviz.plot_pitch()
        ball_is_not_there = plot_ball
        title = "Player, and Side at each intervals"
        if use_annotation:
            title = "Player, and Frame at each intervals"

        for i, (player, side) in enumerate(zip(players, sides)):
            x = f"{side}-P_{player}-x"
            y = f"{side}-P_{player}-y"

            # Prepare player positions and their coordinates on the pitch
            data_x = pd.to_numeric(dataset[x], errors='coerce')
            data_y = pd.to_numeric(dataset[y], errors='coerce')

            # Create DataFrames with a single column for x and y coordinate of a player
            positions_x = pd.DataFrame(data_x, columns=[x])
            positions_y = pd.DataFrame(data_y, columns=[y])

            # Changing the dataset to metric or pitch coordinates
            positions_x = mio.to_metric_coordinates(positions_x)
            positions_y = mio.to_metric_coordinates(positions_y)

            color = self.color_map[i % len(self.color_map)]

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
            ba_x = pd.to_numeric(dataset["ball-x"], errors='coerce')
            ba_y = pd.to_numeric(dataset["ball-y"], errors='coerce')

            ba_x = mio.to_metric_coordinates(pd.DataFrame(ba_x, columns=["ball-x"]))
            ba_y = mio.to_metric_coordinates(pd.DataFrame(ba_y, columns=["ball-y"]))

            ax.plot(ba_x["ball-x"], ba_y["ball-y"], marker=".", linestyle="-.", markersize=int(marker_size * 1.5),
                    color="black", zorder=2, label="Ball")
            ax.plot(ba_x["ball-x"].iloc[0], ba_y["ball-y"].iloc[0], marker=".", linestyle="-.",
                    markersize=int(marker_size * 1.75), color="green", zorder=2)
            ax.plot(ba_x["ball-x"].iloc[-1], ba_y["ball-y"].iloc[-1], marker=".", linestyle="-.",
                    markersize=int(marker_size * 1.75), color="red", zorder=2)

            # Ball movement arrows
            for j in range(len(ba_x) - 1):
                dx = ba_x["ball-x"].iloc[j + 1] - ba_x["ball-x"].iloc[j]
                dy = ba_y["ball-y"].iloc[j + 1] - ba_y["ball-y"].iloc[j]
                if abs(dx) > 0.1 or abs(dy) > 0.1:  # Only draw arrows for significant movements
                    ax.arrow(ba_x["ball-x"].iloc[j], ba_y["ball-y"].iloc[j], dx, dy, head_width=1.2, head_length=1.2,
                             fc='black', ec='black', linestyle="dotted", length_includes_head=True)

        # ax.annotate("Start Point", (ba_x["ball-x"].iloc[0], ba_y["ball-y"].iloc[0]), xytext=(5, 5), textcoords="offset points", fontsize=8, color="black")
        # ax.annotate("End Point", (ba_x["ball-x"].iloc[-1], ba_y["ball-y"].iloc[-1]), xytext=(5, 5), textcoords="offset points", fontsize=8, color="black")

        ball_is_not_there = False

        start_seconds = dataset["Time[s]"].iloc[0]
        end_seconds = dataset["Time[s]"].iloc[-1]
        pl = [i for i in players]
        ax.set_title(
            f"Tracking for Player {pl} in the First Half from {start_seconds} seconds to {end_seconds} seconds")
        ax.legend(title=title, loc="upper left", fontsize=8)

        plt.tight_layout()
        plt.show()

game_2_pre = PreProcessing(data_home="Sample_Game_2_RawTrackingData_Home_Team.csv",
                           data_away="Sample_Game_2_RawTrackingData_Away_Team.csv")
game_2_data = game_2_pre.load_and_process_data(add_ball=False)