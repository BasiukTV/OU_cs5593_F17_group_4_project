from abc import ABC, abstractmethod

class Modeling(ABC):
    """Abstract base class for different implementations of contributor clustering and repository regression model creation."""

    def __init__(self, preproc_dataset_path):
        self.preproc_dataset_path = preproc_dataset_path
        super().__init__()

    @abstractmethod
    def run_modeling(cross_validation_params, model_output_path):
        pass
