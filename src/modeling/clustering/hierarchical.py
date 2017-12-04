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
    """
        Strictly speaking clustering doesn't result in a model, it results in an assignment of a cluster ID
        to every contributor (a.k.a "clustering"). However, as hierarchical clustering uses O(n**2) memory
        and GitHub has over 30M contributors classic approach will not work. So, instead, we cluster
        sufficiently large representative sample of contributors and record its cluster dendrogram cut.
        To cluster contributors outside the sample (or outside the dataset entirely) we take the cut
        and introduce new contributor as a new cluster. Because we use sum group distances, that new cluster
        will be merged to one of the old clusters, because it's so small and will introduce smallest error growth.
        We then repeat this process for all 30M contributors. Problem with this approach is that adding majority
        of data to the clustering after the fact makes our initial dendrogram "untrue", and we basically making
        a claim (hopufuly justified) that if new contributor was a member of original small sample it would be
        assigned to the same cluster, and furthermore, remaining clustering will not change for other contributors.
        We accept this drawback, so we can actually cluster our whole dataset.
    """

    def __init__(self, cluster_sizes, clusters_avg_contributor, weights, sum_cluster_error):
        """Default constructor. Saves relevant clustering parameters."""
        super().__init__()
        self.cluster_sizes = cluster_sizes
        self.clusters_avg_contributor = clusters_avg_contributor
        self.weights = weights
        self.sum_cluster_error = sum_cluster_error

    def cluster_contributor(self, avg_contributor_record):
        """Using self clustering parameters and given contributor average record finds best cluster assignment."""
        closest_clusterID = 0 # Initial closest cluster guess
        closest_distance = self.clusters_avg_contributor[closest_clusterID].eucld_dist(
            avg_contributor_record, self.weights, self_cluster_size=self.cluster_sizes[closest_clusterID])

        # Check group distance to other clusters
        for cID in range(1, len(self.clusters_avg_contributor)):
            candidate_distance = self.clusters_avg_contributor[cID].eucld_dist(
                avg_contributor_record, self.weights, self_cluster_size=self.cluster_sizes[cID])

            if candidate_distance < closest_distance:
                closest_distance = candidate_distance
                closest_clusterID = cID

        return closest_clusterID + 1 # Cluster IDs are 1-indexed

    def serialize_to_file(self, path_to_model):
        # TODO Implement this
        pass

    def deserialize_from_file(self, path_to_model):
        # TODO Implement this
        pass

    def __str__(self):
        """String representation override."""
        result = "This Hierarchical Model is a dendrogram with {} clusters:".format(len(self.cluster_sizes))
        for i in range(len(self.cluster_sizes)):
            result += "\nCluster #{}: Size - {}, Avg. {}".format(
                i + 1, self.cluster_sizes[i], self.clusters_avg_contributor[i])
        result += "\nSum Error of Clusters: {}".format(self.sum_cluster_error)
        return result

class HierarchicalModeling(Modeling):

    def __init__(self, preproc_dataset_path):
        super().__init__(preproc_dataset_path)

    def run_modeling(self, cross_validation_params, weights, output_clustering_db_path):
        # TODO Cross Validation Parameters are strictly speaking only aply for supervised learning. We need to rename this.

        import random
        from copy import copy
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

        log("Retrieving the list of all available contibutor IDs from the dataset.")
        contributor_IDs = db.get_contributor_IDs()
        con_num = len(contributor_IDs) # Number of available contributors
        log("Retrieved {} unique contributor IDs.".format(con_num))

        trial_num = 1 # TODO Configure this, run multiple trials
        trial_size = cross_validation_params.trial_size
        log("Trial #{} : Starting clustering with {} contributors.".format(trial_num, trial_size))

        # Sample (with no replacement, as it slows things down) contributors for the trial.
        trial_contributorIDs = random.sample(contributor_IDs, trial_size)
        trial_weights = weights
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
        p = 0 # Progress counter
        out_of = (len(proximity_matrix) ** 2) / 2 # Progress untill finish
        progress(p, out_of)

        avg_contrib_copy = copy(trial_avg_contributions)
        best_model = None

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
            avg_contrib_copy[min_dist_from_clusterID] = calc_avg_contribution(new_cluster_contributor_records)

            # Iterate through remaining rows of proximity matrix and update them
            for cID1 in proximity_matrix.keys():
                if cID1 == min_dist_from_clusterID: # Distances update for this row will be done bu other iterations
                    continue

                # If this cluster contains distance to min_dist_to_clusterID remove it
                if min_dist_to_clusterID in proximity_matrix[cID1][1].keys():
                    del proximity_matrix[cID1][1][min_dist_to_clusterID]

                # If this cluster contains distance to new cluster update it
                if cID1 in proximity_matrix[min_dist_from_clusterID][1].keys():
                    proximity_matrix[min_dist_from_clusterID][1][cID1] = avg_contrib_copy[min_dist_from_clusterID].eucld_dist(
                        other=avg_contrib_copy[cID1],
                        weights=trial_weights,
                        self_cluster_size=len(proximity_matrix[min_dist_from_clusterID][0]),
                        other_cluster_size=len(proximity_matrix[cID1][0]))
                    continue

                # Otherwise new cluster contains distance to this distance, update it
                proximity_matrix[cID1][1][min_dist_from_clusterID] = avg_contrib_copy[min_dist_from_clusterID].eucld_dist(
                        other=avg_contrib_copy[cID1],
                        weights=trial_weights,
                        self_cluster_size=len(proximity_matrix[min_dist_from_clusterID][0]),
                        other_cluster_size=len(proximity_matrix[cID1][0]))

            # As we going through joining last 5 cluster, let's pick one of the clusterings to be used for "modeling"
            if len(proximity_matrix) <= 5:
                cluster_sizes = []
                clusters_avg_contributor = []
                sum_cluster_error = 0

                # Calculate cluster sizes, average members and dum of errors
                for cID in proximity_matrix:
                    cluster_sizes.append(len(proximity_matrix[cID][0]))
                    clusters_avg_contributor.append(avg_contrib_copy[cID])

                    for contrib in proximity_matrix[cID][0]:
                        sum_cluster_error += avg_contrib_copy[cID].eucld_dist(trial_avg_contributions[contrib])

                candidate_model = HierarchicalModel(cluster_sizes, clusters_avg_contributor, trial_weights, sum_cluster_error)
                if len(proximity_matrix) == 5:
                    progress(1, 1, last=True)
                    log("Now doing merging of final 5 clusters.")
                    best_model = candidate_model
                    continue

                if candidate_model.sum_cluster_error / best_model.sum_cluster_error > 1.1:
                    # If last joing increased sum error of clustering by more than 10%, it's probably a good place to stop.
                    break

                # If error increase was below 10% pick candidate model as best one
                best_model = candidate_model
                continue

            p += len(proximity_matrix) + 1
            progress(p, out_of)

        log("Trial #{} : Done.".format(trial_num))
        log("Hierarchical clustering of sample is done.")

        if output_clustering_db_path:
            log("Will 'model' whole contributor dataset clustering and record it to '{}'".format(output_clustering_db_path))
            from database.sqlite.clustering_db import SQLiteClusteringDatabase
            out_db = SQLiteClusteringDatabase(output_clustering_db_path)

            # For each known contributor ID, find clustering and record it to the out_db
            p = 0 # Progress counter
            for cID in contributor_IDs:
                contributor_averages = db.get_contributor_weekly_averages(cID)
                out_db.record_clustering(cID, best_model.cluster_contributor(contributor_averages))

                p += 1
                progress(p, con_num)

            progress(1, 1, last=True)
            out_db.close()

        db.close()

        return best_model

if  __name__ == "__main__":
    import argparse
    from collections import namedtuple

    # Configuring CLI arguments parser and parsing the arguments
    parser = argparse.ArgumentParser("Script for creating a kmeans hierarchical model of GitHub contributors.")
    parser.add_argument("-d", "--dataset", required=True, help="Path to preprocessed dataset.")
    parser.add_argument("-ts", "--trial-size", type=int, required=True, help="Number of contributors to include in a trial. Hierarchical clustering uses O(n**2) of memory. This number cannot be very large.")
    parser.add_argument("-cdbp", "--clustering-db-path", help="Path to the output clustering db.")
    parser.add_argument("-ws", "--weights", nargs='+', type=float, default=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], help="Array of weights to be aplied to contributor attributes for purposes of determining distances between them.")
    args = parser.parse_args()

    # We use simple named tuple so we don't have to define a class which will not be used anywhere else
    RuntimeParameters = namedtuple('RuntimeParameters', ['trial_size'])

    # TODO Below Is simply a test of imports. Actualy implement the modeling invocation.
    modeling = HierarchicalModeling(args.dataset)
    best_model = modeling.run_modeling(RuntimeParameters(args.trial_size), args.weights, args.clustering_db_path)
    print("Best model is:\n{}".format(best_model))
