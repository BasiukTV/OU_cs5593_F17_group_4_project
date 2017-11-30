from abc import ABC, abstractmethod

class PreprocessingDatabase(ABC):
    """Abstact Base Class of the database instance used for hosting preprocessed data."""

    def __init__(self, endpoint):
        self.endpoint = endpoint
        super().__init__()

    @abstractmethod
    def close(self):
        """Closes database."""
        pass

    @abstractmethod
    def insert_repository_record(self, repository_record):
        """Inserts a repository record into preprocessing database."""
        pass

    @abstractmethod
    def insert_contributor_record(self, contributor_record):
        """Inserts a repository record into preprocessing database."""
        pass

    @abstractmethod
    def get_contributor_IDs(self):
        """Return list of contributors for whom there's data in the dataset."""
        pass

    @abstractmethod
    def get_contributor_weekly_averages(self, contributorID):
        """Returns Contributor record which holds average weekly contributions for given contributorID. """
        pass

    @abstractmethod
    def merge_in_intermidiate_db(self, intermidiate_preprocessing_db):
        """Merges data from an intermidiate preprocessing database into this database."""
        pass

    @abstractmethod
    def serialize_to_file(self, path, fileType="native"):
        """Serializes data currently in this database into a file of specified format (if supported)."""
        pass
