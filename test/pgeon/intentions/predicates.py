from enum import Enum, auto


class PredicateName(Enum):
    HELD_PLAYER = auto()
    HELD_PARTNER = auto()
    PLAYER_POSITION = auto()
    PARTNER_POSITION = auto()
    ACTION2NEAREST = auto()
    POT_STATE = auto()
    DIRECTION = auto()
    AGENT = auto()
    OBJECT = auto()
    ORIENTATIONS = auto()


class Held(Enum):
    NOTHING = auto()
    ONION = auto()
    TOMATO = auto()
    DISH = auto()
    SOUP = auto()


class PotState(Enum):
    FINISHED = auto()
    COOKING = auto()
    PREPARING = auto()
    NOT_STARTED = auto()


class Action2Nearest(Enum):
    LEFT = auto()
    RIGHT = auto()
    TOP = auto()
    BOTTOM = auto()
    STAY = auto()
    INTERACT = auto()


class Direction(Enum):
    NORTH = auto()
    NORTHEAST = auto()
    EAST = auto()
    SOUTHEAST = auto()
    SOUTH = auto()
    SOUTHWEST = auto()
    WEST = auto()
    NORTHWEST = auto()


class Agent(Enum):
    PLAYER = auto()
    PARTNER = auto()


class Object(Enum):
    ONION = auto()
    TOMATO = auto()
    SOUP = auto()
    DISH = auto()
    SERVICE = auto()
    POT0 = auto()
    POT1 = auto()


class Orientations(Enum):
    Top = (0, -1)
    Bottom = (0, 1)
    Right = (1, 0)
    Left = (-1, 0)
