import abc
from typing import Any, Optional, Tuple


class Environment(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "reset")
            and callable(subclass.reset)
            and hasattr(subclass, "step")
            and callable(subclass.step)
        )

    def __init__(self):
        pass

    # TODO Set seed
    @abc.abstractmethod
    def reset(self, seed: Optional[int] = None) -> Any:
        pass

    @abc.abstractmethod
    def step(self, action) -> Tuple[Any, Any, bool, Any]:
        pass
