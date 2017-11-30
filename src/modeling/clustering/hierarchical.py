import os, sys

# Below allows importing our application modules from anywhere under src/ directory where __init__.py file exists
# TODO Below is dirty and probably not how things should be
app_home_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + "/../../..")
app_src_dir = os.path.realpath(app_home_dir + "/src")
sys.path.insert(0, app_src_dir)

from modeling.modeling import Modeling
from modeling.clustering.cluster_model import ClusterModel

from utils.logging import log

class HierarchicalModel(ClusterModel):
    def __init__(self):
        # TODO This simply calls super constructor and might need (or not) updates
        super().__init__()

    def cluster_contributor(self, contributor_record):
        # TODO Implement this
        pass

    def serialize_to_file(self, path_to_model):
        # TODO Implement this
        pass

    def deserialize_from_file(self, path_to_model):
        # TODO Implement this
        pass

class HierarchicalModeling(Modeling):

    def __init__(self, preproc_dataset_path):
        super().__init__(preproc_dataset_path)
        # TODO This simply calls super constructor and might need (or not) updates

    def run_modeling(self, cross_validation_params):
        from database.sqlite.preproc_db_sqlite import SQLitePreprocessingDatabase

        """
            Altough at the moment our preprocesssed dataset is created with src/preproc/aggregate.py,
            which doesn't use any abstractions for setting up the DB, it appears to be compatible with
            SQLitePreprocessingDatabase.

            TODO This needs to be fixed.
        """
        db = SQLitePreprocessingDatabase(endpoint = self.preproc_dataset_path, existing = True)

        # TODO Below line is just a test of accesing DB. Remove it.
        log(db.db_connection.cursor().execute("SELECT * FROM sqlite_master").fetchall())

        db.close()

        # TODO Finish this
        return HierarchicalModel()

if  __name__ == "__main__":
    import argparse

    # Configuring CLI arguments parser and parsing the arguments
    parser = argparse.ArgumentParser("Script for creating a kmeans hierarchical model of GitHub contributors.")
    parser.add_argument("-d", "--dataset", help="Path to preprocessed dataset.")
    args = parser.parse_args()

    # TODO Below Is simply a test of imports. Actualy implement the modeling invocation.
    modeling = HierarchicalModeling(args.dataset)
    model = modeling.run_modeling("not_actual_cross_validation_params")
    model.serialize_to_file("not_an_anctual_path_to_file")
