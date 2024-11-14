from dataclasses import dataclass


@dataclass
class Config:
    """Configuration settings for data preprocessing."""
    COLOR_MAP: list[str] = (
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
    )
    BALL_POSSESSION_SPEED: float = 0.05
    MPS_TO_MPH: float = 2.23694
    PLAYER_NAMES: list[str] = ("P_1", "P_2", "P_3", "P_4", "P_5", "P_6", "P_7", "P_8", "P_9", "P_10", "P_11", "P_25",
                               "P_15", "P_16", "P_17", "P_18", "P_19", "P_20", "P_21", "P_22", "P_23", "P_24")
