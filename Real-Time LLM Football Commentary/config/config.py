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

