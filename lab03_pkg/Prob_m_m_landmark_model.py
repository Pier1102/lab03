import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from  matplotlib.patches import Arc
from Sensors_Models.utils import compute_p_hit_dist
# Import the module and the most relevant functions
import sympy
sympy.init_printing(use_latex='mathjax')
from sympy import symbols, Matrix, latex

arrow = u'$\u2191$'

def landmark_range_bearing_sensor(robot_pose, landmark, sigma, max_range=6.0, fov=math.pi/2):
    """""
    Simulate the detection of a landmark with a virtual sensor able to estimate range and bearing
    """""
    m_x, m_y = landmark[:]
    x, y, theta = robot_pose[:]

    r_ = math.dist([x, y], [m_x, m_y]) + np.random.normal(0., sigma[0])
    phi_ = math.atan2(m_y - y, m_x - x) - theta + np.random.normal(0., sigma[1])

    # filter z for a more realistic sensor simulation (add a max range distance and a FOV)
    if r_ > max_range or abs(phi_) > fov / 2:
        return None

    return [r_, phi_]


def landmark_model_sample_pose(z, landmark, sigma):
    """""
    Sample a robot pose from the landmark model
    Inputs:
        - z: the measurements features (range and bearing of the landmark from the sensor) [r, phi]
        - landmark: the landmark position in the map [m_x, m_y]
        - sigma: the standard deviation of the measurement noise [sigma_r, sigma_phi]
    Outputs:
        - x': the sampled robot pose [x', y', theta']
    """""
    m_x, m_y = landmark[:]
    sigma_r, sigma_phi = sigma[:]

    gamma_hat = np.random.uniform(0, 2*math.pi)
    r_hat = z[0] + np.random.normal(0, sigma_r)
    phi_hat = z[1] + np.random.normal(0, sigma_phi)

    x_ = m_x + r_hat * math.cos(gamma_hat)
    y_ = m_y + r_hat * math.sin(gamma_hat)
    theta_ = gamma_hat - math.pi - phi_hat

    return np.array([x_, y_, theta_])


def plot_sampled_poses(robot_pose, z, landmark, sigma):
    """""
    Plot sampled poses from the landmark model
    """""
    # plot samples poses
    for i in range(1000):
        x_prime = landmark_model_sample_pose(z, landmark, sigma)
        # plot robot pose
        rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
        rotated_marker._transform = rotated_marker.get_transform().rotate_deg(math.degrees(x_prime[2])-90)
        plt.scatter(x_prime[0], x_prime[1], marker=rotated_marker, s=80, facecolors='none', edgecolors='b')
    
    # plot real pose
    rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
    rotated_marker._transform = rotated_marker.get_transform().rotate_deg(math.degrees(robot_pose[2])-90)
    plt.scatter(robot_pose[0], robot_pose[1], marker=rotated_marker, s=140, facecolors='none', edgecolors='r')

    plt.xlabel("x-position [m]")
    plt.ylabel("y-position [m]")
    plt.title("Landmark Model Pose Sampling")
    plt.show()


x,y,theta,mx, my = symbols("x y theta m_x m_y ")
# h_x fuction of the sensor model
hx = Matrix(
    [
        [sympy.sqrt((mx - x) ** 2 + (my - y) ** 2)],
        [sympy.atan2(my - y, mx - x) - theta],
    ]
)
eval_hx = sympy.lambdify((x, y, theta, mx, my), hx, "numpy")
Ht = hx.jacobian(Matrix([x, y, theta])) 
eval_Ht = sympy.lambdify((x, y, theta, mx, my), Ht, "numpy") # function of the Jacobian Ht

def main():
    # robot pose
    robot_pose = np.array([1., 1., math.pi/4])
    x=robot_pose[0]
    y=robot_pose[1]
    theta=robot_pose[2]

    # landmarks position in the map
    landmark = np.array([5., 2.])
   
    # landmark sensor parameters
    fov = math.pi/3 #field of view
    max_range = 6.0 
    sigma = np.array([0.3, math.pi/24]) #uncertainty (range measurement, bearing measurement)


    
    z = landmark_range_bearing_sensor(robot_pose, landmark, sigma)

    # plot landmark
    plt.plot(landmark[0], landmark[1], "sk", ms=10)
    plot_sampled_poses(robot_pose, z, landmark, sigma)
    
    plt.close('all')
    print(f'Jacobian Ht \n: {eval_Ht(x,y,theta,landmark[0],landmark[1])}\n')

if __name__ == "__main__":
    main()