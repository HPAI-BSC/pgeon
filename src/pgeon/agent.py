import abc


class Agent(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, "act") and callable(subclass.act)

    @abc.abstractmethod
    def act(self, state):
        pass
