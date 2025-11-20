import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from landmark_msgs.msg import LandmarkArray
from lab03_pkg.Jacobians import eval_gux, eval_Gt, eval_Vt, eval_hx, eval_Ht
import tf_transformations

class EKF_Robot(Node):
    def __init__(self):
        super().__init__('ekf_node')

        self.dim_x = 3 # x, y, theta
        self.dim_u = 2 # linear velocity and angular velocity
        self.eval_gux=eval_gux
        self.eval_Gt=eval_Gt
        self.eval_Vt=eval_Vt
        self.eval_hx = eval_hx
        self.eval_Ht = eval_Ht
        self.frequency = 20. #Hz

        #initial velocities
        self.v = 0.0
        self.w = 0.0

        # === Known landmark positions (manual definition) ===
        self.landmarks = {
            11: [-1.1,  -1.1, 0.5],
            12: [-1.1,  0.0, 1.5],
            13: [-1.1, 1.1, 0.5],
            21: [0.0,  -1.1, 1.0],
            22: [0.0,  0.0, 0.75],
            23: [0.0, 1.1, 0.3],
            31: [1.1,  -1.1, 1.5],
            32: [1.1,  0.0, 1.0],
            33: [1.1, 1.1, 0.0],
        }

        self.get_logger().info(f"Loaded {len(self.landmarks)} static landmarks from table.")

        """
        Initializes the extended Kalman filter creating the necessary matrices
        """
        #self.mu = np.array([-2.00227,-0.49999,0.])  # mean state estimate inizializzata prendendo i valori da ground truth
        self.mu = np.array([0.0, 0.0, 0.0])  # mean state estimate inizializzata a zero
        
        self.Sigma = np.eye(self.dim_x)  # covariance state estimate

        self.Mt = np.eye(self.dim_u)  # process noise

        self._I = np.eye(self.dim_x)  # identity matrix used for computations
    
    #=== PUB/SUB ===
        self.odom_sub= self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.landmarks_sub = self.create_subscription(LandmarkArray, '/landmarks', self.update_callback,10)
        self.ekf_pub = self.create_publisher(Odometry, '/ekf', 10)

    #=== TIMER ===
        self.timer = self.create_timer(1.0 / self.frequency, self.prediction_callback)
     
    def odom_callback(self, msg):

        self.v = float(msg.twist.twist.linear.x)

        self.w = float(msg.twist.twist.angular.z)
        
    def prediction_callback(self):
        dt = 1.0 / self.frequency
        v = self.v
        w = self.w

        sigma_u = np.array([0.1, 0.1])
        self.Mt = np.diag(sigma_u**2)

        # === CASE 1: straight motion ===
        if abs(w) < 1e-5:

            x, y, theta = self.mu

            # prediction
            x_p = x + v * dt * np.cos(theta)
            y_p = y + v * dt * np.sin(theta)
            theta_p = theta

            self.mu = np.array([x_p, y_p, theta_p], dtype=float)

            # Jacobians
            Gt = np.array([
                [1, 0, -v * dt * np.sin(theta)],
                [0, 1,  v * dt * np.cos(theta)],
                [0, 0,  1]
            ])

            Vt = np.array([
                [dt * np.cos(theta), 0],
                [dt * np.sin(theta), 0],
                [0, dt]
            ])

        # === CASE 2: general unicycle model ===
        else:

            self.mu = np.array(
                self.eval_gux(self.mu[0], self.mu[1], self.mu[2], v, w, dt),
                dtype=float
            ).flatten()

            Gt = np.array(self.eval_Gt(self.mu[0], self.mu[1], self.mu[2], v, w, dt), dtype=float)
            Vt = np.array(self.eval_Vt(self.mu[0], self.mu[1], self.mu[2], v, w, dt), dtype=float)

        # Update covariance
        self.Sigma = Gt @ self.Sigma @ Gt.T + Vt @ self.Mt @ Vt.T

        self.publish_ekf()
       
    def update_callback(self, msg):
        """
        EKF update step based on landmark range and bearing measurements.
        Each landmark measurement provides [range, bearing] relative to the robot.
        """

        # Measurement noise covariance (range, bearing)
        Q = np.diag([0.1, 0.1])  

        # === Loop over all detected landmarks ===
        for landmark in msg.landmarks:
            lm_id = landmark.id
            z = np.array([landmark.range, landmark.bearing])

            # Skip if this ID is not in our known map
            if lm_id not in self.landmarks:
                continue

            # Known landmark coordinates from YAML
            m_x, m_y, m_z = self.landmarks[lm_id]
            z_s = 0.0   # 0.3 nel robot reale , altezza telecamera

            # === Predicted measurement ===
            # Expected [range_hat, bearing_hat] from current estimate μ and landmark m
            z_hat = np.array(
                self.eval_hx(self.mu[0], self.mu[1], self.mu[2], m_x, m_y, m_z, z_s),
                dtype=float
            ).flatten()  #eval hx calculates range and bearing attesi

            # === Measurement Jacobian ===
            
            Ht = np.array(
                self.eval_Ht(self.mu[0], self.mu[1], self.mu[2], m_x, m_y,m_z,z_s),dtype=float)

            # === Innovation (residual) ===
            y = z - z_hat
            # Normalize the bearing angle
            y[1] = np.arctan2(np.sin(y[1]), np.cos(y[1]))

            # === Innovation covariance ===
            S = Ht @ self.Sigma @ Ht.T + Q

            # === Kalman gain ===
            K = self.Sigma @ Ht.T @ np.linalg.inv(S)

            # === Update mean and covariance ===
            self.mu = self.mu + K @ y
            I = np.eye(3)
            self.Sigma = (I - K @ Ht) @ self.Sigma

            self.get_logger().info(f"mu dopo update:\n{self.mu}")
            self.get_logger().info(f"Sigma dopo update:\n{self.Sigma}")

            self.publish_ekf()

    def publish_ekf(self):
        # Create odometry message
        msg = Odometry()

        # Header
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"     # Global reference frame
        msg.child_frame_id = "base_footprint"  # Robot frame

        # === 1. Mean state μ ===
        x = float(self.mu[0])
        y = float(self.mu[1])
        theta = float(self.mu[2])

        # Position
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0   # robot is on the ground

        # Orientation (yaw → quaternion)
        q = tf_transformations.quaternion_from_euler(0.0, 0.0, theta)
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]

        # === 2. Fill covariance as 6×6 ===
        Sigma = self.Sigma   # your 3×3 EKF covariance

        # Build the 6×6 covariance for ROS
        cov = np.zeros((6,6))

        # Fill x, y, yaw parts
        cov[0,0] = Sigma[0,0]   # σxx
        cov[0,1] = Sigma[0,1]   # σxy
        cov[1,0] = Sigma[1,0]   # σyx
        cov[1,1] = Sigma[1,1]   # σyy

        cov[0,5] = Sigma[0,2]   # σxθ
        cov[1,5] = Sigma[1,2]   # σyθ
        cov[5,0] = Sigma[2,0]   # σθx
        cov[5,1] = Sigma[2,1]   # σθy
        cov[5,5] = Sigma[2,2]   # σθθ

        # Mark z, roll, pitch as HIGH uncertainty (unknown)
        cov[2,2] = 99999.0      # z
        cov[3,3] = 99999.0      # roll
        cov[4,4] = 99999.0      # pitch

        # Flatten to row-major (expected by ROS)
        msg.pose.covariance = cov.flatten().tolist() # questa riga è obbligatoria per inviare correttamente la covarianza in un Odometry.

        # Publish
        self.ekf_pub.publish(msg)


def main():
    rclpy.init()
    node = EKF_Robot()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()