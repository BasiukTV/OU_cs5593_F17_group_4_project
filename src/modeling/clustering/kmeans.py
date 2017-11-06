import os, sys

# Below allows importing our application modules from anywhere under src/ directory where __init__.py file exists
# TODO Below is dirty and probably not how things should be
app_home_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + "/../../..")
app_src_dir = os.path.realpath(app_home_dir + "/src")
sys.path.insert(0, app_src_dir)

from modeling.modeling import Modeling
from modeling.clustering.cluster_model import ClusterModel

class KMeansModel(ClusterModel):
    def __init__(self):
        # TODO This simply calls super constructor and might need (or not) updates
        super().__init__()

    def cluster_contributor(sefl, repository_record):
        # TODO Implement this
        pass

    def serialize_to_file(self, path_to_model):
        # TODO Implement this
        pass

    def initialize_from_file(self, path_to_model):
        # TODO Implement this
        pass

class KMeansModeling(Modeling):

    def __init__(self, preproc_dataset_path):
        super().__init__(preproc_dataset_path)
        # TODO This simply calls super constructor and might need (or not) updates

    def run_modeling(cross_validation_params, model_output_path):
        # TODO Implement this
        return KMeansModel()

if  __name__ == "__main__":
    import argparse

    # Configuring CLI arguments parser and parsing the arguments
    parser = argparse.ArgumentParser("Script for creating a kmeans clustering model of GitHub contributors.")
    parser.add_argument("-d", "--dataset", help="Path to preprocessed dataset.")
    args = parser.parse_args()

    # TODO Below Is simply a test of imports. Actualy implement the modeling invocation.
    modeling = KMeansModeling(args.dataset)
    model = modeling.run_modeling()
    model.serialize_to_file("not_an_anctual_path_to_file")
    pass