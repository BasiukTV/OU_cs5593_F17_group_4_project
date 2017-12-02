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
        import random
        from database.sqlite.preproc_db_sqlite import SQLitePreprocessingDatabase

        log("Starting hierarchical clustering of the contributors.")
        log("Connecting to the dataset at '{}'".format(self.preproc_dataset_path))

        """
            Altough at the moment our preprocesssed dataset is created with src/preproc/aggregate.py,
            which doesn't use any abstractions for setting up the DB, it appears to be compatible with
            SQLitePreprocessingDatabase.

            TODO This needs to be fixed.
        """
        db = SQLitePreprocessingDatabase(endpoint = self.preproc_dataset_path, existing = True)

        log("Retrieving the list of available contibutor IDs from the dataset.")
        contributor_IDs = db.get_contributor_IDs()
        con_num = len(contributor_IDs) # Number of available contributors
        log("Retrieved {} unique contributor IDs.".format(con_num))

        trial_num = 1
        trial_size = 20 # TODO Make this configurable
        log("Trial #{} : Starting clustering with {} contributors.".format(trial_num, trial_size))

        # Sample (with no replacement, as it slows things down) contributors for the trial.
        trial_contributorIDs = random.sample(contributor_IDs, trial_size)
        trial_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        trial_avg_contributions = {}

        log("Trial #{} : Calculating contribution averages.".format(trial_num))
        for c_ID in trial_contributorIDs:
            trial_avg_contributions[c_ID] = db.get_contributor_weekly_averages(c_ID)

        log("Trial #{} : Building initial proximity matrix.".format(trial_num))
        proximity = {}
        for c1_ID in trial_contributorIDs:
            proximity[c1_ID] = {}
            for c2_ID in trial_contributorIDs:
                if c1_ID == c2_ID:
                    proximity[c1_ID][c2_ID] = 0
                    continue

                proximity[c1_ID][c2_ID] = trial_avg_contributions[c1_ID].eucld_dist(
                    trial_avg_contributions[c2_ID],
                    weights=trial_weights)

        log("Trial #{} : Done.".format(trial_num))
        log("Hierarchical clustering is done.")

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
