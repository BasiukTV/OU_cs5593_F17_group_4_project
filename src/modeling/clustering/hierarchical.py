import os, sys

# Below allows importing our application modules from anywhere under src/ directory where __init__.py file exists
# TODO Below is dirty and probably not how things should be
app_home_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + "/../../..")
app_src_dir = os.path.realpath(app_home_dir + "/src")
sys.path.insert(0, app_src_dir)

from modeling.modeling import Modeling
from modeling.clustering.cluster_model import ClusterModel
from preproc.record.contributor import calc_avg_contribution

from utils.logging import log, progress

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
        trial_size = 100 # TODO Make this configurable
        log("Trial #{} : Starting clustering with {} contributors.".format(trial_num, trial_size))

        # Sample (with no replacement, as it slows things down) contributors for the trial.
        trial_contributorIDs = random.sample(contributor_IDs, trial_size)
        trial_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        trial_avg_contributions = {}

        log("Trial #{} : Calculating contribution averages.".format(trial_num))
        for i in range(trial_size):
            c_ID = trial_contributorIDs[i]
            trial_avg_contributions[c_ID] = db.get_contributor_weekly_averages(c_ID)

            progress(i, trial_size)
        progress(trial_size, trial_size, last=True)

        log("Trial #{} : Building initial proximity matrix.".format(trial_num))
        proximity_matrix = {}
        for i in range(trial_size):
            c1_ID = trial_contributorIDs[i]
            proximity_matrix[c1_ID] = ({c1_ID}, {}) # ({members of cluster}, {distances to other clusters})

            # We will only build upper triangular part of the matrix as it is symetric and 0-diagonal.
            for j in range(i + 1, trial_size):
                c2_ID = trial_contributorIDs[j]

                proximity_matrix[c1_ID][1][c2_ID] = trial_avg_contributions[c1_ID].eucld_dist(
                    trial_avg_contributions[c2_ID],
                    weights=trial_weights)

        log("Trial #{} : Starting to merge clusters in.".format(trial_num))
        while len(proximity_matrix) > 1:
            # Arbitrarily pick one distance between clusters as a min distance

            """
                Below relies on iterator picking top row of the proximity matrix.
                If it picks bottom row second iterator will fail to retieve next item.

                TODO Fix this.
            """
            min_dist_from_clusterID = next(iter(proximity_matrix.keys()))
            min_dist_to_clusterID = next(iter(proximity_matrix[min_dist_from_clusterID][1].keys()))

            # Now find actual minimal distance betweem clusters
            for i in proximity_matrix.keys():
                for j in proximity_matrix[i][1].keys():
                    if proximity_matrix[i][1][j] < proximity_matrix[min_dist_from_clusterID][1][min_dist_to_clusterID]:
                        min_dist_from_clusterID = i
                        min_dist_to_clusterID = j

            # Move all members of min_dist_to_clusterID into min_dist_from_clusterID
            proximity_matrix[min_dist_from_clusterID] = (proximity_matrix[min_dist_from_clusterID][0].union(
                proximity_matrix[min_dist_to_clusterID][0]), proximity_matrix[min_dist_from_clusterID][1])

            # Remove row of proximity_matrix corresponding to min_dist_to_clusterID
            del proximity_matrix[min_dist_to_clusterID]
            # Remove min_dist_to_clusterID entry from proximity_matrix[min_dist_from_clusterID][1]
            del proximity_matrix[min_dist_from_clusterID][1][min_dist_to_clusterID]

            # Prepare new cluster average contribution
            new_cluster_contributor_records = []
            for cID in proximity_matrix[min_dist_from_clusterID][0]:
                new_cluster_contributor_records.append(trial_avg_contributions[cID])
            new_cluster_average_contribution = calc_avg_contribution(new_cluster_contributor_records)

            # Iterate through remaining rows of proximity matrix and update them
            for cID1 in proximity_matrix.keys():
                if cID1 == min_dist_from_clusterID: # Distances update for this row will be done bu other iterations
                    continue

                # If this cluster contains distance to min_dist_to_clusterID remove it
                if min_dist_to_clusterID in proximity_matrix[cID1][1].keys():
                    del proximity_matrix[cID1][1][min_dist_to_clusterID]

                # Prepare this cluster average contribution
                # TODO This cluster membership didn't change, we don't need to recalculate its average contribution
                cluster_contributor_records = []
                for cID2 in proximity_matrix[cID1][0]:
                    cluster_contributor_records.append(trial_avg_contributions[cID2])
                cluster_average_contribution = calc_avg_contribution(cluster_contributor_records)

                # If this cluster contains distance to new cluster update it
                if cID1 in proximity_matrix[min_dist_from_clusterID][1].keys():
                    proximity_matrix[min_dist_from_clusterID][1][cID1] = new_cluster_average_contribution.eucld_dist(
                        other=cluster_average_contribution,
                        weights=trial_weights,
                        self_cluster_size=len(proximity_matrix[min_dist_from_clusterID][0]),
                        other_cluster_size=len(proximity_matrix[cID1][0]))
                    continue

                # Otherwise new cluster contains distance to this distance, update it
                proximity_matrix[cID1][1][min_dist_from_clusterID] = new_cluster_average_contribution.eucld_dist(
                        other=cluster_average_contribution,
                        weights=trial_weights,
                        self_cluster_size=len(proximity_matrix[min_dist_from_clusterID][0]),
                        other_cluster_size=len(proximity_matrix[cID1][0]))

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
