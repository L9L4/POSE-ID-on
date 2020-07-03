import numpy as np
import random
from tqdm import tqdm

class K_Means:
    """
    K_means algorithm, with distance function customized for dealing with angular values
    (https://en.wikipedia.org/wiki/Mean_of_circular_quantities)
    """

    def __init__(self, k=2, tol=0.0001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def diff(self, l1, l2):
      diffs =  []
      for i in range(len(l1)):
        distance = 1-np.cos(l1-l2)
        diffs.append(distance)
      return np.mean(diffs)
    
    def angular_mean(self, feature):
      feature = np.array(feature)
      mean = []
      for f in range(feature.shape[1]):
        col = feature[:,f]
        x = []
        y = []
        for angle in col:
            x.append(np.cos(angle))
            y.append(np.sin(angle))
        x = np.median(x)
        y = np.median(y)
        arc = np.arctan2(y, x)
        if arc < 0:
          arc = arc+2*np.pi
        mean.append(arc)
      return np.array(mean)

    def fit(self,data):

        self.centroids = {}
        self.samples = random.sample(range(len(data)), self.k)

        for i in range(self.k):
            self.centroids[i] = data[self.samples[i]]

        for j in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [self.diff(featureset,self.centroids[centroid]) for centroid in self.centroids]  ##
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = self.angular_mean(self.classifications[classification]) ##

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if self.diff(original_centroid,current_centroid) > self.tol:
                    optimized = False

            if optimized:
                break

        return self.centroids, self.classifications