# simulation.py
import numpy as np
import scipy.integrate as scy_int
from config import TF, SATURATION
from models import DCMotor
from controllers import PID_controller
from utils import Metrics, get_ref

motor_instance = None

def setup_motor(a, k):
    global motor_instance
    motor_instance = DCMotor(a, k, SATURATION)

def connected_systems_model(t, states, tau_ref_func, data_dict, pid_params):
    tau, _ = states 
    ref = tau_ref_func(t, "ramp")
    dt = t - data_dict["prev_t"]
    
    u_control = PID_controller(tau, ref, *pid_params, dt, data_dict)
    
    data_dict["prev_t"] = t
    
    taup = motor_instance.taup_by_model(tau, u_control)
    return [taup, 0] 

def run_simulation_cost(pid_params):
    kp, ki, kd = pid_params

    if kp < 0 or ki < 0 or kd < 0: return 1e6 
    
    t_eval = np.linspace(0, TF, int(TF/0.05))
    data_dict = {"integral": 0, "prev_err": 0, "prev_t": 0}
    
    try:
        res = scy_int.solve_ivp(
            connected_systems_model, [0, TF], [0, 0],
            args=(get_ref, data_dict, pid_params),
            t_eval=t_eval, method='RK45'
        )
        m = Metrics(get_ref(res.t, "ramp"), res.y[0])
        # Return the metric to be minimized
        return m.itae()[-1]
    except:
        return 1e12