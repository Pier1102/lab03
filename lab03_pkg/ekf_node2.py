import rclpy
from rclpy.node import Node
import numpy as np

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from landmark_msgs.msg import LandmarkArray
import tf_transformations

from lab03_pkg.Jacobians import (
    eval_gux,
    eval_Gt,
    eval_Vt,
    eval_hx_landmark,
    eval_Ht_landmark,
    eval_hx_odom,
    eval_Ht_odom,
    eval_hx_imu,
    eval_Ht_imu,
)


class EKF_Robot(Node):
    def __init__(self):
        super().__init__("ekf_node2")

        # stato [x, y, theta, v, w]
        self.dim_x = 5
        self.dim_u = 2
        self.frequency = 20.0
        self.dt = 1.0 / self.frequency

        # funzioni simboliche
        self.eval_gux = eval_gux
        self.eval_Gt = eval_Gt
        self.eval_Vt = eval_Vt
        self.eval_hx_landmark = eval_hx_landmark
        self.eval_Ht_landmark = eval_Ht_landmark
        self.eval_hx_odom = eval_hx_odom
        self.eval_Ht_odom = eval_Ht_odom
        self.eval_hx_imu = eval_hx_imu
        self.eval_Ht_imu = eval_Ht_imu

        # stato iniziale [x, y, theta, v, w]
        self.mu = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.Sigma = np.eye(self.dim_x, dtype=float)

        # rumori
        self.sigma_u = np.array([0.05, 0.05], dtype=float)  #  noise su [v_cmd, w_cmd]
        self.Mt = np.diag(self.sigma_u**2)

        self.std_odom = np.array([0.1, 0.1], dtype=float)   # misura [v, w]
        self.std_imu = np.array([0.05], dtype=float)        # misura [w]

        # comandi usati nel modello di predizione
        self.v_cmd = 0.0
        self.w_cmd = 0.0

        # landmark con z (come task 1)
        self.landmarks = {
            11: [-1.1, -1.1, 0.5],
            12: [-1.1, 0.0, 1.5],
            13: [-1.1, 1.1, 0.5],
            21: [0.0, -1.1, 1.0],
            22: [0.0, 0.0, 0.75],
            23: [0.0, 1.1, 0.3],
            31: [1.1, -1.1, 1.5],
            32: [1.1, 0.0, 1.0],
            33: [1.1, 1.1, 0.0],
        }
        self.get_logger().info(f"Loaded {len(self.landmarks)} landmarks")

        # pub/sub
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.imu_sub = self.create_subscription(Imu, "/imu", self.imu_callback, 10)
        self.landmarks_sub = self.create_subscription(
            LandmarkArray, "/landmarks", self.update_callback, 10
        )
        self.ekf_pub = self.create_publisher(Odometry, "/ekf", 10)

        # timer di predizione
        self.timer = self.create_timer(self.dt, self.prediction_callback)

    # ======================================================
    # PREDICTION STEP
    # ======================================================
    def prediction_callback(self):
        x, y, theta, v, w = self.mu
        v_cmd = self.v_cmd
        w_cmd = self.w_cmd
        dt = self.dt

        # motion model
        g = np.array(self.eval_gux(x, y, theta, v, w, v_cmd, w_cmd, dt),dtype=float).flatten()

        Gt = np.array(self.eval_Gt(x, y, theta, v, w, v_cmd, w_cmd, dt),dtype=float)
        Vt = np.array(self.eval_Vt(x, y, theta, v, w, v_cmd, w_cmd, dt),dtype=float)

        self.mu = g
        self.Sigma = Gt @ self.Sigma @ Gt.T + Vt @ self.Mt @ Vt.T

        self.publish_ekf()

    # ======================================================
    # UPDATE ODOM: misura [v, w]
    # ======================================================
    def odom_callback(self, msg: Odometry):
        v_meas = float(msg.twist.twist.linear.x)
        w_meas = float(msg.twist.twist.angular.z)

        # uso queste come comandi del modello per lo step successivo
        self.v_cmd = v_meas
        self.w_cmd = w_meas

        z = np.array([v_meas, w_meas], dtype=float)
        Q = np.diag(self.std_odom**2)

        x, y, theta, v, w = self.mu

        z_hat = np.array(self.eval_hx_odom(x, y, theta, v, w),dtype=float).flatten()

        H = np.array(self.eval_Ht_odom(x, y, theta, v, w),dtype=float)

        y_res = z - z_hat

        S = H @ self.Sigma @ H.T + Q
        K = self.Sigma @ H.T @ np.linalg.inv(S)

        self.mu = self.mu + K @ y_res
        I = np.eye(self.dim_x)
        self.Sigma = (I - K @ H) @ self.Sigma

    # ======================================================
    # UPDATE IMU: misura [w]
    # ======================================================
    def imu_callback(self, msg: Imu):
        w_meas = float(msg.angular_velocity.z)

        z = np.array([w_meas], dtype=float)
        Q = np.diag(self.std_imu**2)

        x, y, theta, v, w = self.mu

        z_hat = np.array(self.eval_hx_imu(x, y, theta, v, w),dtype=float).flatten()

        H = np.array(self.eval_Ht_imu(x, y, theta, v, w),dtype=float)

        y_res = z - z_hat

        S = H @ self.Sigma @ H.T + Q
        K = self.Sigma @ H.T @ np.linalg.inv(S)

        self.mu = self.mu + K @ y_res
        I = np.eye(self.dim_x)
        self.Sigma = (I - K @ H) @ self.Sigma

    # ======================================================
    # UPDATE LANDMARK: misura [range, bearing]
    # ======================================================
    def update_callback(self, msg: LandmarkArray):
        Q = np.diag([0.1, 0.1])  # noise su [range, bearing]
        z_s = 0.0  # altezza sensore in simulazione

        for landmark in msg.landmarks:
            lm_id = landmark.id
            if lm_id not in self.landmarks:
                continue

            m_x, m_y, m_z = self.landmarks[lm_id]
            z = np.array([landmark.range, landmark.bearing], dtype=float)

            x, y, theta, v, w = self.mu

            z_hat = np.array(self.eval_hx_landmark(x, y, theta, v, w, m_x, m_y, m_z, z_s),dtype=float).flatten()

            H = np.array(self.eval_Ht_landmark(x, y, theta, v, w, m_x, m_y, m_z, z_s), dtype=float)

            y_res = z - z_hat
            y_res[1] = np.arctan2(np.sin(y_res[1]), np.cos(y_res[1]))

            S = H @ self.Sigma @ H.T + Q
            K = self.Sigma @ H.T @ np.linalg.inv(S)

            self.mu = self.mu + K @ y_res
            I = np.eye(self.dim_x)
            self.Sigma = (I - K @ H) @ self.Sigma

            self.get_logger().info(f"mu dopo update:\n{self.mu}")
            self.get_logger().info(f"Sigma dopo update:\n{self.Sigma}")


        self.publish_ekf()

    # ======================================================
    # PUBLISH EKF STATE AS ODOM
    # ======================================================
    def publish_ekf(self):
        x, y, theta, v, w = self.mu
        Sigma = self.Sigma     # 5×5

        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.child_frame_id = "base_footprint"

        # ---- POSITION ----
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.position.z = 0.0

        # ---- ORIENTATION ----
        q = tf_transformations.quaternion_from_euler(0.0, 0.0, float(theta))
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]

        # ======================================================
        #  POSE COVARIANCE (6x6)
        #  only x,y,theta are included in the pose covariance
        # ======================================================

        pose_cov = np.zeros((6, 6))

        pose_cov[0,0] = Sigma[0,0]   # x-x
        pose_cov[0,1] = Sigma[0,1]
        pose_cov[1,0] = Sigma[1,0]
        pose_cov[1,1] = Sigma[1,1]   # y-y

        pose_cov[0,5] = Sigma[0,2]
        pose_cov[1,5] = Sigma[1,2]
        pose_cov[5,0] = Sigma[2,0]
        pose_cov[5,1] = Sigma[2,1]

        pose_cov[5,5] = Sigma[2,2]   # theta-theta

        # z, roll, pitch unknown
        pose_cov[2,2] = 99999.0
        pose_cov[3,3] = 99999.0
        pose_cov[4,4] = 99999.0

        msg.pose.covariance = pose_cov.flatten().tolist()

        #  TWIST COVARIANCE (6x6)

        twist_cov = np.zeros((6, 6))

        # v in twist.linear.x --> row 0
        twist_cov[0,0] = Sigma[3,3]     # var(v)

        # w in twist.angular.z --> row 5
        twist_cov[5,5] = Sigma[4,4]     # var(w)

        msg.twist.covariance = twist_cov.flatten().tolist()

        # ---- TWIST VALUES ----
        msg.twist.twist.linear.x = float(v)
        msg.twist.twist.angular.z = float(w)

        self.ekf_pub.publish(msg)

def main():
    rclpy.init()
    node = EKF_Robot()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()