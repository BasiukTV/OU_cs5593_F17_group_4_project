# Standard library imports
import sqlite3

class SQLiteClusteringDatabase(object):
    """SQLIte class for the database instance used for hosting preprocessed data."""

    def __init__(self, endpoint, existing=False):
        import os

        super().__init__() # Call to a super constructor

        # Check whether file exists if connecting to an existing database
        if existing == True:
            if not os.path.isfile(endpoint):
                raise Exception("{} file doesn't exist. Cannot load the database.".format(endpoint))
        else:
            # If new database is requested ...
            try:
                """If file already exists at the path. Remove that file."""
                os.remove(endpoint)
            except:
                # A file not existing at the path, is what we normaly expect.
                pass

        # Establish an SQLite connection
        self.db_connection = sqlite3.connect(endpoint)

        # Install a schema for a new preprocessing database
        if existing == False:
            cursor = self.db_connection.cursor()

            # Create a table for repository preprocessed data
            cursor.execute("""
                CREATE TABLE contributor_clustering (
                    contributorID integer,
                    clusterID integer
                )""")

            # Create an index on contributorID
            cursor.execute("CREATE INDEX contributorIdx ON contributor_clustering(contributorID)")

            self.db_connection.commit()

    def close(self):
        """Closes database."""
        self.db_connection.close()

    def record_clustering(self, contributorID, clusterID):
        """Records mapping of a contributorID to clusterID"""
        cursor = self.db_connection.cursor()
        cursor.execute("INSERT INTO contributor_clustering VALUES (?, ?)", (contributorID, clusterID))
        self.db_connection.commit()
