from abc import ABC, abstractmethod

class RegressionModel(ABC):
    """Abstract base class of a model produced by regression modeling with different algorithms."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def regression_on_repository(sefl, repository_record):
        """Returns a regression on given repository_record according to the model."""
        pass

    @abstractmethod
    def serialize_to_file(self, path_to_model):
        """Saves current model to a file."""
        pass

    @abstractmethod
    def initialize_from_file(self, path_to_model):
        """Initializes this model from a file."""
        pass
