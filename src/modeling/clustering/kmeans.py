import os, sys
import math, random
import sqlite3
from sqlite3 import Error

# Below allows importing our application modules from anywhere under src/ directory where __init__.py file exists
# TODO Below is dirty and probably not how things should be
app_home_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + "/../../..")
app_src_dir = os.path.realpath(app_home_dir + "/src")
sys.path.insert(0, app_src_dir)

from modeling.modeling import Modeling

def select_contributors(conn):
    cur = conn.cursor()
    cur.execute('select contributorID, sum(repos_started_count), sum(repos_forked_count), sum(code_pushed_count), sum(pull_request_created_count), sum(pull_request_reviewed_count), sum(issue_created_count), sum(issue_resolved_count), sum(issue_commented_count), sum(issue_other_activity_count), sum(owned_repos_starts_count) from contributor group by contributorID')

    rows = cur.fetchall()

    contributors_data = []    
    for row in rows:
       contributors_data.append(list(row))

    return contributors_data

def getContributorsData(db_file):
    conn = create_db_connection(db_file)
    
    with conn:
        return select_contributors(conn)   
       
def create_db_connection(db_file):
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return None

def update_user_cluster(conn, cluster, contributor_ids):
    cur = conn.cursor()
    
    for cid in contributor_ids:
        cur.execute('''update user set cluster = %d where id = %d''' %(cluster, cid))        

    #formatted_qry = '''update user set cluster = ? where id in ({})'''.format(','.join('?' * len(contributor_ids)))
    #print('formated qry = ', formatted_qry)
    #cur.execute(formatted_qry, (cluster, contributor_ids))

class KMeansModeling(Modeling):

    def __init__(self, preproc_dataset_path, k):
        super().__init__(preproc_dataset_path)
        self.k = k
        
        # stop updating centroids when their differences is less than 0.005
        self.diff = 0.0049
       
        db_file = '../../../samples/data/preproc/sample.sqlite3'

        # get contributors data from database
        self.data_list = getContributorsData(db_file)       

        self.centroids = []
        clusters = do_kmeans(self.data_list, self.k, self.diff)          
        
        conn = create_db_connection(db_file)

        for i in range(len(clusters)):
            with conn:
                update_user_cluster(conn, i+1, [p[0] for p in clusters[i].points])
                #print('#### Cluster %s ########' %i)
                #print(clusters[i].points)  
                #print('###########################################################')                

    def run_modeling(self, cross_validation_params):
        # TODO Implement this        
        pass

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
            smallest_distance = getDistance(p[1:], clusters[0].centroid)

            # set the cluster this point belongs to
            clusterIndex = 0

            # for the rest of the clusters
            for i in range(clusterCount - 1):
                # calculate the distance of that point to each other cluster's centroid
                distance = getDistance(p[1:], clusters[i+1].centroid)
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
        # data except for the one in index 0
        coords = [p[1:] for p in self.points]
        # Reformat that so all x's are together, all y'z, and so on.
        unzipped = zip(*coords)
        # Calculate the mean for each dimension
        centroid_coords = [math.fsum(dList)/numPoints for dList in unzipped]
        # print("Centroid is %s" %(str(centroid_coords)))
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
    parser.add_argument("-k", "--clusters", type=int, help="Chosen k clusters.")
    args = parser.parse_args()
   
    # invoke k-means algorithm
    KMeansModeling(args.dataset, args.clusters)
    #model = modeling.run_modeling("not_actual_cross_validation_params")
    pass
