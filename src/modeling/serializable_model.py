from abc import ABC, abstractmethod

class SerializableModel(ABC):
    """Abstract base class of a data mining model which can be serialized to and desirialized from a file."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def serialize_to_file(self, path_to_model):
        """Saves current model to a file."""
        pass

    @abstractmethod
    def deserialize_from_file(self, path_to_model):
        """Initializes this model from a file."""
        pass
