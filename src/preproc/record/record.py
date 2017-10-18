class Record:
    """Base class for the preprocessed dataset record."""

    def __init__(self, entityID, timestamp):
        """Sets common record attributes."""

        self.entityID = entityID
        self.timestamp = timestamp
