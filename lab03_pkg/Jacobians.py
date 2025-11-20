# import sympy
# from sympy import symbols, Matrix

# # ==========================================================
# # 1. VELOCITY MOTION MODEL JACOBIANS (G_t, V_t)
# # ==========================================================
# x, y, theta, v, w, dt = symbols('x y theta v w dt') 
# R = v / w
# beta = theta + w * dt

# gux = Matrix([
#     [x - R * sympy.sin(theta) + R * sympy.sin(beta)],
#     [y + R * sympy.cos(theta) - R * sympy.cos(beta)],
#     [beta],
#     ])

# eval_gux = sympy.lambdify((x, y, theta, v, w, dt), gux, 'numpy')
# Gt = gux.jacobian(Matrix([x, y, theta]))
# eval_Gt = sympy.lambdify((x, y, theta, v, w, dt), Gt, "numpy")
# Vt = gux.jacobian(Matrix([v, w]))
# eval_Vt = sympy.lambdify((x, y, theta, v, w, dt), Vt, "numpy")

# # ==========================================================
# # 2. LANDMARK SENSOR MODEL JACOBIAN (H_t)
# # ==========================================================
# x, y, theta, mx, my , mz , z_s = symbols("x y theta m_x m_y m_z z_s")
# hx = Matrix([
#     [sympy.sqrt((mx - x) ** 2 + (my - y) ** 2 + (mz - z_s) ** 2)],
#     [sympy.atan2(my - y, mx - x) - theta],
# ])

# eval_hx = sympy.lambdify((x, y, theta, mx, my, mz,z_s), hx, "numpy")
# Ht = hx.jacobian(Matrix([x, y, theta]))
# eval_Ht = sympy.lambdify((x, y, theta, mx, my,mz,z_s), Ht, "numpy")

# # ==========================================================
# # Exported symbols
# # ==========================================================
# __all__ = ['eval_gux', 'eval_Gt', 'eval_Vt', 'eval_hx', 'eval_Ht']
import sympy
from sympy import symbols, Matrix

# ==========================================================
# 1. MOTION MODEL g(x, u)
# stato: [x, y, theta, v, w]
# comandi: [v_cmd, w_cmd]
# ==========================================================

x, y, theta, v, w, v_cmd, w_cmd, dt = symbols("x y theta v w v_cmd w_cmd dt")

gux = Matrix([
    x + v * dt * sympy.cos(theta),
    y + v * dt * sympy.sin(theta),
    theta + w * dt,
    v_cmd,
    w_cmd
])

eval_gux = sympy.lambdify(
    (x, y, theta, v, w, v_cmd, w_cmd, dt),gux,"numpy")

Gt = gux.jacobian(Matrix([x, y, theta, v, w]))

eval_Gt = sympy.lambdify((x, y, theta, v, w, v_cmd, w_cmd, dt),Gt,"numpy")

Vt = gux.jacobian(Matrix([v_cmd, w_cmd]))
eval_Vt = sympy.lambdify((x, y, theta, v, w, v_cmd, w_cmd, dt),Vt,"numpy")

# ==========================================================
# 2. LANDMARK SENSOR MODEL h_landmark(x)
# misura: [range, bearing]
# stato: [x, y, theta, v, w]
# landmark: [m_x, m_y, m_z], altezza sensore z_s
# ==========================================================

mx, my, mz, z_s = symbols("m_x m_y m_z z_s")

hx_landmark = Matrix([
    sympy.sqrt((mx - x) ** 2 + (my - y) ** 2 + (mz - z_s) ** 2),
    sympy.atan2(my - y, mx - x) - theta])

eval_hx_landmark = sympy.lambdify((x, y, theta, v, w, mx, my, mz, z_s),hx_landmark,"numpy")

Ht_landmark = hx_landmark.jacobian(Matrix([x, y, theta, v, w]))

eval_Ht_landmark = sympy.lambdify((x, y, theta, v, w, mx, my, mz, z_s),Ht_landmark,"numpy")

# ==========================================================
# 3. ODOM MODEL h_odom(x)
# misura: [v, w]
# ==========================================================

h_odom = Matrix([v,w])

eval_hx_odom = sympy.lambdify((x, y, theta, v, w),h_odom,"numpy")

Ht_odom = h_odom.jacobian(Matrix([x, y, theta, v, w]))

eval_Ht_odom = sympy.lambdify((x, y, theta, v, w),Ht_odom,"numpy")

# ==========================================================
# 4. IMU MODEL h_imu(x)
# misura: [w]
# ==========================================================

h_imu = Matrix([w])

eval_hx_imu = sympy.lambdify( (x, y, theta, v, w),h_imu,"numpy")

Ht_imu = h_imu.jacobian(Matrix([x, y, theta, v, w]))

eval_Ht_imu = sympy.lambdify((x, y, theta, v, w),Ht_imu,"numpy")

# Export

__all__ = [
    "eval_gux",
    "eval_Gt",
    "eval_Vt",
    "eval_hx_landmark",
    "eval_Ht_landmark",
    "eval_hx_odom",
    "eval_Ht_odom",
    "eval_hx_imu",
    "eval_Ht_imu",
]