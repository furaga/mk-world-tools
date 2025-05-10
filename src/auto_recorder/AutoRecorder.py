from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


class Player:
    def __init__(self, name: str, rate: int):
        self.name = name
        self.rate = rate


class MatchInfo:
    def __init__(self, players: list[Player], course: str, rule: str):
        self.players = players
        self.course = course
        self.rule = rule


class ResultInfo:
    def __init__(self, players: list[Player], my_rate: int, my_place: int):
        self.players = players
        self.my_rate = my_rate
        self.my_place = my_place


class AutoRecorder(ABC):
    @abstractmethod
    def detect_match_info(self, img: np.ndarray) -> Tuple[bool, MatchInfo]:
        """
        マッチングしたプレイヤー情報やこれから走るコース・ルールを検出する
        """
        pass

    @abstractmethod
    def detect_result(
        self, img: np.ndarray, match_info: MatchInfo
    ) -> Tuple[bool, ResultInfo]:
        """
        レース結果画面を検出する
        """
        pass
