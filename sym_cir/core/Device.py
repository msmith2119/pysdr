

from abc import ABC, abstractmethod

class Device(ABC):
    """
    Abstract base class for all circuit elements.
    """

    def __init__(self, name: str):
        self.name = name

    @property
    @abstractmethod
    def nodes(self) -> tuple[int, ...]:
        """Tuple of node numbers used by this device."""
        pass

    @abstractmethod
    def stamp(self, builder):
        """Stamp this device into the MNA matrix."""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"