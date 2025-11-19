import sympy
from sympy import symbols, Matrix

# ==========================================================
# 1. VELOCITY MOTION MODEL JACOBIANS (G_t, V_t)
# ==========================================================
x, y, theta, v, w, dt = symbols('x y theta v w dt') 
R = v / w
beta = theta + w * dt

gux = Matrix([
    [x - R * sympy.sin(theta) + R * sympy.sin(beta)],
    [y + R * sympy.cos(theta) - R * sympy.cos(beta)],
    [beta],
])

eval_gux = sympy.lambdify((x, y, theta, v, w, dt), gux, 'numpy')
Gt = gux.jacobian(Matrix([x, y, theta]))
eval_Gt = sympy.lambdify((x, y, theta, v, w, dt), Gt, "numpy")
Vt = gux.jacobian(Matrix([v, w]))
eval_Vt = sympy.lambdify((x, y, theta, v, w, dt), Vt, "numpy")

# ==========================================================
# 2. LANDMARK SENSOR MODEL JACOBIAN (H_t)
# ==========================================================
x, y, theta, mx, my , mz , z_s = symbols("x y theta m_x m_y m_z z_s")
hx = Matrix([
    [sympy.sqrt((mx - x) ** 2 + (my - y) ** 2 + (mz - z_s) ** 2)],
    [sympy.atan2(my - y, mx - x) - theta],
])

eval_hx = sympy.lambdify((x, y, theta, mx, my, mz,z_s), hx, "numpy")
Ht = hx.jacobian(Matrix([x, y, theta]))
eval_Ht = sympy.lambdify((x, y, theta, mx, my,mz,z_s), Ht, "numpy")

# ==========================================================
# Exported symbols
# ==========================================================
__all__ = ['eval_gux', 'eval_Gt', 'eval_Vt', 'eval_hx', 'eval_Ht']