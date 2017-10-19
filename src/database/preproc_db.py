from abc import ABC, abstractmethod

class PreprocessingDatabase(ABC):
    """Abstact Base Class of the database instance used for hosting preprocessed data."""

    def __init__(self, endpoint, existing=False):
        self.endpoint = endpoint
        super().__init__()

    @abstractmethod
    def insert_repository_record(repository_record):
        """Inserts a repository record into preprocessing database."""
        pass

    @abstractmethod
    def insert_contributor_record(contributor_record):
        """Inserts a repository record into preprocessing database."""
        pass

    @abstractmethod
    def merge_in_intermidiate_db(intermidiate_preprocessing_db):
        """Merges data from an intermidiate preprocessing database into this database."""
        pass

    @abstractmethod
    def serialize_to_file(path, fileType="native"):
        """Serializes data currently in this database into a file of specified format (if supported)."""
        pass