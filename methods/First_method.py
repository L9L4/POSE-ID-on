import numpy as np
from scipy.optimize import fmin
from tqdm import tqdm
import operator
from utils.utils import *


class MatchingClass1:

    """
    Matching Class Method 1.
    """
    
    links = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[1,8],[7,6],[8,9],[8,12],
             [9,10],[10,11],[12,13],[13,14]]
    weight = 1  
    
    def __init__(self, pose_a, pose_b):
        self.pose_a = pose_a
        self.pose_b = pose_b
        
    def coefficients(self, joint_1_a, joint_2_a, joint_1_b, joint_2_b):

        """
        Compute auxiliary coefficients starting from 4 points: joint_1_a, joint_2_a, joint_1_b, joint_2_b.
        """
    
        x_1a = joint_1_a[0]
        y_1a = joint_1_a[1]
    
        x_2a = joint_2_a[0]
        y_2a = joint_2_a[1]
        
        x_1b = joint_1_b[0]
        y_1b = joint_1_b[1]
    
        x_2b = joint_2_b[0]
        y_2b = joint_2_b[1]
    
        u = x_1a - x_2a
        t = y_1a - y_2a
    
        s = x_1b - x_2b
        p = y_1b - y_2b
    
        alpha = s*u + p*t
        beta = - p*u + s*t
        gamma = np.sqrt((u**2 + t**2)*(s**2 + p**2))
    
        delta = alpha/gamma
        epsilon = beta/gamma
    
        return delta, epsilon

    def omega_i(self, omega, i):
        
        """
        Compute the angle omega_i between the i-th limbs of two different poses, 
        having previously rotated the second pose by an angle omega.
        """

        delta, epsilon = self.coefficients(self.pose_a[self.links[i][0]], 
                                           self.pose_a[self.links[i][1]], 
                                           self.pose_b[self.links[i][0]], 
                                           self.pose_b[self.links[i][1]])
        return np.arccos(delta*np.cos(omega) + epsilon*np.sin(omega))

    def function(self, omega):
        
        """
        Sum all the angles omega_i obtained by comparing all the limbs of the two poses, 
        having previously rotated the second pose by an angle omega.
        """
        
        function = 0*omega
    
        for i in range(len(self.links)):
            function += self.omega_i(omega, i)*self.weight
    
        return function

    def rotation(self, x, omega):
        
        """
        Rotate the pose x by an angle omega
        """
        
        x0, y0 = x.T[0], x.T[1]
        c, s = np.cos(omega), np.sin(omega)
        x1 = c*x0 - s*y0
        y1 = s*x0 + c*y0
        x_1 = np.array([x1, y1])
        return x_1 
    
    def minimum(self):
        
        """
        Get the optimal value omega_star of omega by which the two poses are most similar, and
        then compute the difference (loss) according to that omega.
        """
        
        omega_star = fmin(self.function, 0, disp=False)[0]
        loss = self.function(omega_star)
        return omega_star, loss

def first_method_app(dict_joints_SR_destrorso, mirroring = False, turning = False):
    
    """
    Apply the first method to search for the most similar poses to a given one (within the dictionary dict_joints_SR_destrorso).
    Mirroring and turning can be selected.
    Output:
        best_worst_cases_1: dictionary with:
            keys: poses of the input dictionary
            values: list with the 5 closest poses (inluding the query) and the farthest one
    """
    
    best_worst_cases_1 = {}
    for i in tqdm(range(len(dict_joints_SR_destrorso))):
        sample = dict_joints_SR_destrorso[list(dict_joints_SR_destrorso.keys())[i]]
        samples = [sample]
        if mirroring == True:
            sample_mirr = pose_mirroring(sample)
            samples.append(sample_mirr)
        if turning == True:
            sample_turn = turn_pose(sample)
            samples.append(sample_turn)

        dict_losses = {}
        for key in list(dict_joints_SR_destrorso.keys()):  
            losses = [MatchingClass1(s, dict_joints_SR_destrorso[key]).minimum()[1] for s in samples]
            dict_losses[key] = np.min(losses)

        sorted_d = sorted(dict_losses.items(), key=operator.itemgetter(1))
        best_5 = sorted_d[:5]
        worst = sorted_d[-1]

        best_worst_cases_1[i] = [best_5, worst]
    return best_worst_cases_1
