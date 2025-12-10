# controllers.py
import numpy as np
from config import SATURATION

def PID_controller(tau, tau_ref, kp, ki, kd, dt, err_dict):
    error = tau_ref - tau
    
    # Anti-Windup
    if dt > 0:
        err_dict["integral"] += error * dt
        err_dict["integral"] = np.clip(err_dict["integral"], -SATURATION, SATURATION)
        derivative_error = (error - err_dict["prev_err"]) / dt
    else:
        derivative_error = 0

    out = (kp * error) + (ki * err_dict["integral"]) + (kd * derivative_error)
    err_dict["prev_err"] = error

    return np.clip(out, -SATURATION, SATURATION)
