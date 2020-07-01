import numpy as np
from scipy.optimize import fmin


class MatchingClass1:
    
    links = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[1,8],[7,6],[8,9],[8,12],
             [9,10],[10,11],[12,13],[13,14]]
    weight = 1  
    
    def __init__(self, pose_a, pose_b):
        self.pose_a = pose_a
        self.pose_b = pose_b
        
    def coefficients(self, joint_1_a, joint_2_a, joint_1_b, joint_2_b):
    
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
        delta, epsilon = self.coefficients(self.pose_a[self.links[i][0]], 
                                           self.pose_a[self.links[i][1]], 
                                           self.pose_b[self.links[i][0]], 
                                           self.pose_b[self.links[i][1]])
        return np.arccos(delta*np.cos(omega) + epsilon*np.sin(omega))

    def function(self, omega):
        function = 0*omega
    
        for i in range(len(self.links)):
            function += self.omega_i(omega, i)*self.weight
    
        return function

    def rotation(self, x, omega):
        x0, y0 = x.T[0], x.T[1]
        c, s = np.cos(omega), np.sin(omega)
        x1 = c*x0 - s*y0
        y1 = s*x0 + c*y0
        x_1 = np.array([x1, y1])
        return x_1 
    
    def minimum(self):
        omega_star = fmin(self.function, 0, disp=False)[0]
        loss = self.function(omega_star)
        return omega_star, loss