from abc import ABC, abstractmethod

class ClusterModel(ABC):
    """Abstract base class of a model produced by cluster modeling with different algorithms."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def cluster_contributor(sefl, contibutor_record):
        """Returns cluster identifier given contibutor_record belongs to according to the model."""
        pass

    @abstractmethod
    def serialize_to_file(self, path_to_model):
        """Saves current model to a file."""
        pass

    @abstractmethod
    def initialize_from_file(self, path_to_model):
        """Initializes this model from a file."""
        pass
