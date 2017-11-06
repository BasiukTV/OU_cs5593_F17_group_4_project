from abc import abstractmethod
from modeling.serializable_model import SerializableModel

class ClusterModel(SerializableModel):
    """Abstract base class of a model produced by cluster modeling with different algorithms."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def cluster_contributor(sefl, contibutor_record):
        """Returns cluster identifier (integer from 1 to 5) given contibutor_record belongs to according to the model."""
        pass
