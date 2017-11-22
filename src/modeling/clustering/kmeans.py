import os, sys
import math, random

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
        
        # stop updating centroids when their differences is less than 0.005
        self.diff = 0.0049
        # list of number of contributor clusters - to choose the adequate k from
        self.k_list = [2, 3, 4, 5]
        # a list of each contributor's data
        # TODO get the preprocessed and aggregated data for each contributor
        self.data_list = [[]]
 
    def do_kmeans_clustering(self):
        # for each given k, do k-means clustering
        for k in self.k_list:
            clusters = do_kmeans(self.data_list, k, self.diff)
            #TODO visualization of the clusters    

    def run_modeling(self, cross_validation_params):
        # TODO Implement this
        return KMeansModel()

def do_kmeans(points, k, diff):
    # k random points to use as our initial centroids
    centroids = random.sample(points, k)

    # k clusters using those centroids
    clusters = [Cluster([p]) for p in centroids]

    # loop until the clusters don't change
    counter = 0
    while True:
        # a list of lists to hold the points in each cluster
        lists =[[] for _ in clusters]
        clusterCount = len(clusters)
    
        counter += 1
        for p in points:
            # distance between the point and the centroid of the first cluster
            smallest_distance = getDistance(p, clusters[0].centroid)

            # set the cluster this point belongs to
            clusterIndex = 0

            # for the rest of the clusters
            for i in range(clusterCount - 1):
                # calculate the distance of that point to each other cluster's centroid
                distance = getDistance(p, clusters[i+1].centroid)
                # if closer to that cluster's centroid then
                if distance < smallest_distance:
                    smallest_distance = distance
                    clusterIndex = i+1
            # set the point to belong to that cluster
            lists[clusterIndex].append(p)
           
        # Set biggest_shift to zero for this iteration
        biggest_shift = 0.0
 
        for i in range(clusterCount):
            # calculate how far the centroid moved in this iteration
            shift = clusters[i].update(lists[i])
            # Keep track of the largest move from all cluster centroid updates
            biggest_shift = max(biggest_shift, shift)

        # stop when the centroid doesn't move much
        if biggest_shift < diff:
            break
    return clusters
    
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
