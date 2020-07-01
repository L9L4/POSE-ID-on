import numpy as np
import random
from tqdm import tqdm

class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

     def diff(self, l1, l2):
      diffs =  []
      for i in range(len(l1)):
        diff_1 = np.abs(l1[i] - l2[i])
        if diff_1 <= np.pi:
          diff_2 = diff_1
        else:
          diff_2 = np.min([np.abs(l1[i] - l2[i] - 2*np.pi), np.abs(l1[i] - l2[i] + 2*np.pi)])
        diffs.append(diff_2)
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

        for i in tqdm(range(self.max_iter)):
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
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    # print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break

        return self.centroids, self.classifications