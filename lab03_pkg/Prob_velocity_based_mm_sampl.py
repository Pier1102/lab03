import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import cos, sin, degrees
import matplotlib as mpl
# Import the module and the most relevant functions
import sympy
sympy.init_printing(use_latex='mathjax')
from sympy import symbols, Matrix, latex

arrow = u'$\u2191$'
# ==========================================================
# 1. PROBABILISTIC VELOCITY MOTION MODEL (sampling)
# ==========================================================
def sample_velocity_motion_model(x, u, a, dt):
    """ Sample velocity motion model.
    Arguments:
    x -- pose of the robot before moving [x, y, theta]
    u -- velocity reading obtained from the robot [v, w]
    a -- noise parameters of the motion model [a1, a2, a3, a4, a5, a6]
    dt -- time interval of prediction
    """
    v_hat = u[0] + np.random.normal(0, a[0]*u[0]**2 + a[1]*u[1]**2)
    # u[0] is te componenent of linear velocity  , normal distribution centered in 0 
    w_hat = u[1] + np.random.normal(0, a[2]*u[0]**2 + a[3]*u[1]**2)
    gamma_hat = np.random.normal(0, a[4]*u[0]**2 + a[5]*u[1]**2)
    #add term to take in account the final orientation of the robot

    #From the geometry we can compute the new pose starting from the old one
    r = v_hat/w_hat
    x_prime = x[0] - r*sin(x[2]) + r*sin(x[2]+w_hat*dt)
    y_prime = x[1] + r*cos(x[2]) - r*cos(x[2]+w_hat*dt)
    theta_prime = x[2] + w_hat*dt + gamma_hat*dt

    return np.array([x_prime, y_prime, theta_prime]) #estimated new pose


# ==========================================================
# 3. SYMBOLIC JACOBIANS (G_t, V_t)
# ==========================================================
x, y, theta, v, w, dt = symbols('x y theta v w dt')
R = v / w
beta = theta + w * dt
gux = Matrix(
    [
        [x - R * sympy.sin(theta) + R * sympy.sin(beta)],
        [y + R * sympy.cos(theta) - R * sympy.cos(beta)],
        [beta],
    ]
)

eval_gux = sympy.lambdify((x, y, theta, v, w, dt), gux, 'numpy') #function version of the matrix

Gt = gux.jacobian(Matrix([x, y, theta])) # x y theta are the state variables
eval_Gt = sympy.lambdify((x, y, theta, v, w, dt), Gt, "numpy") #to transform the jacobian in a function


Vt = gux.jacobian(Matrix([v, w])) # jacobian wrt the command
eval_Vt = sympy.lambdify((x, y, theta, v, w, dt), Vt, "numpy")



def plot_velocity_samples(x,u,dt,n_samples,a):
    x_prime= np.zeros([n_samples,3])
    for i in range(n_samples):
        x_prime[i,:] = sample_velocity_motion_model(x, u, a, dt)

    ###################################
    ### Sampling the velocity model ###
    ###################################

    rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
    rotated_marker._transform = rotated_marker.get_transform().rotate_deg(degrees(x[2])-90)
    plt.scatter(x[0], x[1], marker=rotated_marker, s=100, facecolors='none', edgecolors='b')

    for x_ in x_prime[:200]:
        rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
        rotated_marker._transform = rotated_marker.get_transform().rotate_deg(degrees(x_[2])-90)
        plt.scatter(x_[0], x_[1], marker=rotated_marker, s=40, facecolors='none', edgecolors='r')

    plt.xlabel("x-position [m]")
    plt.ylabel("y-position [m]")
    plt.title("velocity motion model sampling")
    plt.savefig("velocity_samples.pdf")
    plt.show()



def main():
    plt.close('all')
    n_samples = 500
    n_bins = 100
    dt = 0.5

    x = [2, 4, 0]# initial pose
    u = [0.8, 0.6] #velocity command (linear and angular velocity)
    a_linear_uncertanty = [0.1, 0.1, 0.001, 0.001, 0.05, 0.05] # noise variance noise we have in every component of robot motion
    a_angular_uncertanty = [0.001, 0.001, 0.1, 0.1, 0.05, 0.05]
    plot_velocity_samples(x,u,dt,n_samples,a_linear_uncertanty)
    plot_velocity_samples(x,u,dt,n_samples,a_angular_uncertanty) 
    print(f'Jacobian Gt \n{eval_Gt(x[0],x[1],x[2],u[0],u[1],dt)}\n')    
    print(f'Jacobian Vt \n{eval_Vt(x[0],x[1],x[2],u[0],u[1],dt)}\n')

if __name__ == "__main__":
    main()
