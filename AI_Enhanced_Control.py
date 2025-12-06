import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from ipywidgets import interact, FloatSlider, IntSlider

def simulate_motor(Kp=15, Ki=0.1, Kd=0.01, theta_target=1.0, ratio=4):
    J = 0.01
    b = 0.1
    K = 0.01
    dt = 0.01
    T = 2
    steps = int(T/dt)

    theta_pid = 0.0
    omega_pid = 0.0
    integral_error_pid = 0.0
    theta_history_pid = []

    for i in range(steps):
        error = theta_target - theta_pid
        integral_error_pid += error*dt
        derivative_error_pid = -omega_pid

        V_pid = Kp*error + Ki*integral_error_pid + Kd*derivative_error_pid

        alpha_motor = (K*V_pid - b*omega_pid)/J
        omega_pid += alpha_motor*dt
        theta_pid += omega_pid*dt
        theta_history_pid.append(theta_pid)

    theta_current_ai_pid = 0.0
    omega_current_ai_pid = 0.0
    integral_error_current_ai_pid = 0.0

    X_online = []
    y_online = []
    theta_history_first_half_ai_pid = []

    for i in range(steps//ratio):
        error = theta_target - theta_current_ai_pid
        integral_error_current_ai_pid += error*dt
        derivative_error = -omega_current_ai_pid

        V_pid_collect = Kp*error + Ki*integral_error_current_ai_pid + Kd*derivative_error

        alpha_motor = (K*V_pid_collect - b*omega_current_ai_pid)/J
        omega_current_ai_pid += alpha_motor*dt
        theta_current_ai_pid += omega_current_ai_pid*dt

        X_online.append([theta_current_ai_pid, theta_target])
        y_online.append(V_pid_collect)
        theta_history_first_half_ai_pid.append(theta_current_ai_pid)

    ai_model_online = LinearRegression()
    ai_model_online.fit(np.array(X_online), np.array(y_online))

    integral_error_online = integral_error_current_ai_pid
    theta_online = theta_current_ai_pid
    omega_online = omega_current_ai_pid

    theta_history_second_half_ai_pid = []

    for i in range(steps//ratio, steps):
        error = theta_target - theta_online
        integral_error_online += error*dt
        derivative_error_online = -omega_online

        V_pid_residual = Kp*error + Ki*integral_error_online + Kd*derivative_error_online
        V_ff = ai_model_online.predict(np.array([[theta_online, theta_target]]))[0]
        V_total = V_ff + 0.3*V_pid_residual

        alpha_motor = (K*V_total - b*omega_online)/J
        omega_online += alpha_motor*dt
        theta_online += omega_online*dt
        theta_history_second_half_ai_pid.append(theta_online)

    theta_history_online_combined = theta_history_first_half_ai_pid + theta_history_second_half_ai_pid

    time = np.arange(steps)*dt

    plt.figure(figsize=(10,5))
    plt.plot(time, theta_history_pid, label="Pure PID")
    plt.plot(time, theta_history_online_combined, label="Online AI + PID")
    plt.plot(time, [theta_target]*steps, '--', color='black', label="Target")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [rad]")
    plt.title(f"DC Motor Response: PID vs Online AI+PID\nKp={Kp}, Ki={Ki}, Kd={Kd}, Target={theta_target}, Ratio={ratio}")
    plt.legend()
    plt.grid(True)
    plt.show()

interact(simulate_motor,
         Kp=FloatSlider(value=15, min=0, max=50, step=1, description='Kp'),
         Ki=FloatSlider(value=0.1, min=0, max=5, step=0.05, description='Ki'),
         Kd=FloatSlider(value=0.01, min=0, max=1, step=0.01, description='Kd'),
         theta_target=FloatSlider(value=1.0, min=-2.0, max=2.0, step=0.05, description='Target'),
         ratio=IntSlider(value=4, min=2, max=10, step=1, description='Learning ratio'));
