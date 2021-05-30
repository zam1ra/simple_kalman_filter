'''
Created on May 30, 2021
@author: Zamira
Simple implementation of a Kalman filter based on:
"Special Topics - The Kalman Filter", Michel van Biezen
https://www.youtube.com/playlist?list=PLX2gX-ftPVXU3oUFNATxGXY90AULiqnWT
'''

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

"""
  A - state transition matrix
  B - control-input matrix applied to the control vector
  w - process noise vector
  Q - covariance of the process noise
  R - covariance of the measurement noise
  C - observation model
  z - measurement noise
  H - transition matrix
  P -  state covariance matrix
  X - state matrix
  Y - measurement of the state
 """
class KalmanFilter(object):

    def __init__(self, A = None, B = None, C = None,
                 Q = None, R = None, H = None, P = None, w = None, z = None, x0 = None):

        if(A is None):
            raise ValueError("Set proper system dynamics.")

        self.n = A.shape[1]
        self.w = 0 if w is None else w
        self.z = 0 if z is None else z

        self.A = A # state transition matrix
        self.C = np.eye(self.n) if C is None else C
        self.B = 0 if B is None else B
        self.Q = 0 if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.H = np.identity(self.n) if H is None else H
        self.X = np.zeros((self.n, 1)) if x0 is None else x0
        self.P = np.eye(self.n) if P is None else P

    def predict(self, u = 0):
        # calculate the predicted state
        self.X = np.dot(self.A, self.X) + np.dot(self.B, u) + self.w
        # update predicted process covariance matrix
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        self.P[0][1]=0
        self.P[1][0]=0
        return self.X, self.P

    def update(self, y):
        # calculate a new observation
        y_new = self.C.dot(y) + self.z
        # calculate kalman gain
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        k = np.dot(np.dot(self.P, self.H), inv(S))
        print(k, '\n\n')
        I = np.eye(self.n)
        #update process covariance
        self.P = np.dot(I - np.dot(k, self.H), self.P)
        # calculate current state
        self.X = self.X + np.dot(k, (y_new - np.dot(self.H, self.X)))
        return self.X, self.P


def example1d(est=50, est_e=4, mea_e=10, mu=70, t_range=1000):
    s = np.random.normal(mu, mea_e, t_range)
    estimation = []
    for mea in s:
        kgain = est_e / (est_e - mea_e)
        est = est + kgain * (mea - est)
        est_e = (1 - kgain) * est_e
        estimation.append(est)
    plt.plot(s, 'r', label='measurments')
    plt.plot(estimation, 'g', label='estimated KF')
    plt.legend()
    plt.show()

def example2d():
    x_observations = np.array([4000, 4260, 4550, 4860, 5110])
    v_observations = np.array([280, 282, 285, 286, 290])
    measurements = np.c_[x_observations, v_observations]
    dt = 1
    A = np.array([[1, dt], [0, 1]])
    B = np.array([[0.5] , [1]])
    w = 0
    a = 2
    # process errors
    xp, vp = 20, 5
    # observation errors
    xm, vm = 3, 2
    # initial process covariance matrix
    P = np.array([[xp*xp, 0], [0, vp*vp]])
    # set covariance of the measurement noise
    R = np.array([[xm*xm, 0], [0, vm*vm]])


    kf = KalmanFilter(A=A, B=B, P=P, R=R, w=w, x0=measurements[0].reshape(2, 1))
    predictions = []
    predictionsKf = []
    for x in measurements[1:]:
        x_pred, p_pred = kf.predict(u=a)
        predictions.append(x_pred[0])
        new_x, new_p = kf.update(x.reshape(2, 1))
        predictionsKf.append(new_x[0])

    plt.plot(range(len(x_observations)), x_observations, label='Measurements')
    plt.plot(range(len(predictions)), np.array(predictions), label='Prediction')
    plt.plot(range(len(predictionsKf)), np.array(predictionsKf), label='Kalman Filter Prediction')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    example1d()
    example2d()