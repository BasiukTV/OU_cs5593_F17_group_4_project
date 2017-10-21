from database.preproc_db import PreprocessingDatabase
import sqlite3

class SQLitePreprocessingDatabase(PreprocessingDatabase):
    """SQLIte class for the database instance used for hosting preprocessed data."""

    def __init__(self, endpoint, existing=False):
        import os

        super().__init__(endpoint) # Call to a super constructor

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
                CREATE TABLE repository (
                    repositoryID integer,
                    timestamp integer,
                    star_count integer,
                    total_contributor_count integer,
                    contributor_type1_count integer,
                    contributor_type2_count integer,
                    contributor_type3_count integer,
                    contributor_type4_count integer,
                    contributor_type5_count integer,
                    code_push_count integer,
                    pull_request_count integer,
                    fork_count integer,
                    release_count integer,
                    active_issues_count integer,
                    resolved_issue_count integer,
                    org_activity_count integer,
                    reserve1 integer,
                    reserve2 text
                )""")

            # Create an index on repositoryID and timestamp
            cursor.execute("CREATE INDEX RepoIdx ON repository(repositoryID, timestamp)")

            # Create a table for contributor preprocessed data
            cursor.execute("""
                CREATE TABLE contributor (
                    contributorID integer,
                    timestamp integer,
                    repos_started_count integer,
                    repos_forked_count integer,
                    code_pushed_count integer,
                    pull_request_created_count integer,
                    pull_request_reviewed_count integer,
                    issue_created_count integer,
                    issue_resolved_count integer,
                    issue_commented_count integer,
                    issue_other_activity_count integer,
                    owned_repos_starts_count integer,
                    reserve1 integer,
                    reserve2 text
                )""")

            # Create an index on contributorID and timestamp
            cursor.execute("CREATE INDEX ContibIdx ON contributor(contributorID, timestamp)")

            self.db_connection.commit();

    def close(self):
        """Closes database."""
        self.db_connection.close()

    def insert_repository_record(self, repository_record):
        """Inserts a repository record into preprocessing database."""
        # TODO Implement this
        pass

    def insert_contributor_record(self, contributor_record):
        """Inserts a repository record into preprocessing database."""
        # TODO Implement this
        pass

    def merge_in_intermidiate_db(self, intermidiate_preprocessing_db):
        """Merges data from an intermidiate preprocessing database into this database."""
        # TODO Implement this
        pass

    def serialize_to_file(self, path, fileType="native"):
        """Serializes data currently in this database into a file of specified format (if supported)."""
        # TODO Implement this
        pass
