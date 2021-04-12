import numpy as np
import random
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage as hc
from scipy.cluster.hierarchy import fcluster as fc

class Clustering:
    
    """
    K-Medians, K-Means and Hierarchical CLustering algorithms are implemented. Both with the 
    euclidean and angular (https://en.wikipedia.org/wiki/Mean_of_circular_quantities) distances
    Parameters:
        k: number of clusters;
        tol: tolerance for the clustering convergence;
        max_iter: maximum number of iterations for the clustering convergence;
        distance: could be 'angular' or 'euclidean';
        method: could be 'K-medians', 'K-means', 'Hierarchical'
    """

    def __init__(self, k=2, tol=0.0001, max_iter=300, distance = 'angular', method = 'K-medians'):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.distance = distance
        self.method = method

    def diff(self, l1, l2):
      diffs =  []
      for i in range(len(l1)):
        distance = 1-np.cos(l1-l2)
        diffs.append(distance)
      return np.mean(diffs)

    def eucl_distance(self, l1, l2):
      return np.linalg.norm(l1-l2)

    def angular_mean(self, feature):
      feature = np.array(feature)
      if len(feature.shape) == 1:
        feature = np.reshape(feature, (1,91))
      mean = []
      for f in range(feature.shape[1]):
        col = feature[:,f]
        x = []
        y = []
        for angle in col:
            x.append(np.cos(angle))
            y.append(np.sin(angle))

        if self.method == 'K-medians':
        	x = np.median(x)
        	y = np.median(y)
        elif self.method == 'K-means':
        	x = np.mean(x)
        	y = np.mean(y)
        arc = np.arctan2(y, x)
        if arc < 0:
          arc = arc+2*np.pi
        mean.append(arc)
      return np.array(mean)

    def eucl_mean(self, feature):
      feature = np.array(feature)
      if len(feature.shape) == 1:
        feature = np.reshape(feature, (1,91))
      if self.method == 'K-medians':
      	M = np.median(features, axis = 0)
      elif self.method == 'K-means':
      	M = np.mean(features, axis = 0)
      return np.array(M)

    def fit(self,data):

    	self.centroids = {}

    	if self.method == 'Hierarchical':
    		if self.distance == 'angular':
    			cl = hc(data, metric = lambda u, v: self.diff(u,v))
    		elif self.distance == 'euclidean':
    			cl = hc(data)
    		else:
    			print('Wrong distance')

    		labels = fc(cl, self.k, criterion = 'maxclust') - 1
    		self.classifications = {k: [] for k in range(self.k)}
    		for i in range(len(data)):
    			self.classifications[labels[i]].append(data[i])

    	else:
    		self.samples = random.sample(range(len(data)), self.k)

    		for i in range(self.k):
    			self.centroids[i] = data[self.samples[i]]

    		for j in range(self.max_iter):
    			self.classifications = {}

    			for i in range(self.k):
    				self.classifications[i] = []

    			for featureset in data:
    				if self.distance == 'angular':
    					distances = [self.diff(featureset,self.centroids[centroid]) for centroid in self.centroids]  ##
    				elif self.distance == 'euclidean':
    					distances = [self.eucl_distance(featureset,self.centroids[centroid]) for centroid in self.centroids]  ##
    				else:
    					print('Wrong distance')

    				classification = distances.index(min(distances))
    				self.classifications[classification].append(featureset)

    			prev_centroids = dict(self.centroids)

    			for classification in self.classifications:
    				if self.distance == 'angular':
    					self.centroids[classification] = self.angular_mean(self.classifications[classification]) ##
    				elif self.distance == 'euclidean':
    					self.centroids[classification] = self.eucl_mean(self.classifications[classification]) ##

    			optimized = True

    			for c in self.centroids:
    				original_centroid = prev_centroids[c]
    				current_centroid = self.centroids[c]
    				if self.distance == 'angular':
    					if self.diff(original_centroid,current_centroid) > self.tol:
    						optimized = False
    				elif self.distance == 'euclidean':
    					if self.eucl_distance(original_centroid,current_centroid) > self.tol:
    						optimized = False

    			if optimized:
    				break

    	return self.centroids, self.classifications