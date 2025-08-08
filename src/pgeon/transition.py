from typing import Any

from pydantic import BaseModel


class Transition(BaseModel):
    action: Any
    probability: float = 0.0
    frequency: int = 0
