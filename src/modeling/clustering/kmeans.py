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

    def cluster_contributor(self, contributor_record):
        # TODO Implement this
        pass

    def serialize_to_file(self, path_to_model):
        # TODO Implement this
        pass

    def deserialize_from_file(self, path_to_model):
        # TODO Implement this
        pass

class KMeansModeling(Modeling):

    def __init__(self, preproc_dataset_path):
        super().__init__(preproc_dataset_path)
        # TODO This simply calls super constructor and might need (or not) updates

    def run_modeling(self, cross_validation_params):
        # TODO Implement this
        return KMeansModel()
    
class Cluster(object):
    ''' A set of points and their centroid '''
    def __init__(self, points):
        # the points that belong to this cluster
        self.points = points
         
        # set up the initial centroid 
        self.centroid = self.calculateCentroid()

    def update(self, points):
        ''' Returns the distance between the previous centroid and 
        the new after recalculating and storing the new centroid '''
    
        old_centroid = self.centroid
        self.points = points
        self.centroid = self.calculateCentroid()
        shift = getDistance(old_centroid, self.centroid)
        return shift

    def calculateCentroid(self):
        ''' Find a virtual center point for a group of n-dimensional points '''
        numPoints = len(self.points)
        # print("Points = %s" % (str(self.points)))
        coords = [p for p in self.points]
        # Reformat that so all x's are together, all y'z, and so on.
        unzipped = zip(*coords)
        # Calculate the mean for each dimension
        centroid_coords = [math.fsum(dList)/numPoints for dList in unzipped]
        #print("Centroid is %s" %(str(centroid_coords)))
        return centroid_coords

def getDistance(a, b):
    ''' calculate Euclidean distance between two points '''

    diff_sum = 0.0
    for i in range(len(a)):
        squared_diff = pow((a[i]-b[i]), 2)
        diff_sum += squared_diff
    dist = math.sqrt(diff_sum)
    return dist    

if  __name__ == "__main__":
    import argparse

    # Configuring CLI arguments parser and parsing the arguments
    parser = argparse.ArgumentParser("Script for creating a kmeans clustering model of GitHub contributors.")
    parser.add_argument("-d", "--dataset", help="Path to preprocessed dataset.")
    parser.add_argument("-k", "--clusters", help="Chosen k clusters.")
    args = parser.parse_args()

    # TODO Below Is simply a test of imports. Actualy implement the modeling invocation.
    modeling = KMeansModeling(args.dataset)
    model = modeling.run_modeling("not_actual_cross_validation_params")
    model.serialize_to_file("not_an_anctual_path_to_file")
    pass
