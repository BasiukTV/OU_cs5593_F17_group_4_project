from abc import abstractmethod
from modeling.serializable_model import SerializableModel

class RegressionModel(SerializableModel):
    """Abstract base class of a model produced by regression modeling with different algorithms."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def regression_on_repository(sefl, repository_record):
        """Returns a regression on given repository_record according to the model."""
        pass
