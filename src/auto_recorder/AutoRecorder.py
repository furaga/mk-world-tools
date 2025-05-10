from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


from pydantic import BaseModel


class Player(BaseModel):
    name: str
    rate: int


class MatchInfo(BaseModel):
    players: list[Player]
    course: str
    rule: str


class ResultInfo(BaseModel):
    players: list[Player]
    my_rate: int
    my_place: int


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
