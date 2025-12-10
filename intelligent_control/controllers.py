import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from config import SATURATION
class FuzzyFactory:
    _simulation_instance = None

    @classmethod
    def get_instance(cls):
        if cls._simulation_instance is None:
            cls._create_system()
        return cls._simulation_instance

    @classmethod
    def _create_system(cls):
        limit_erro = 40
        limit_derro = 70.0
        limit_delta_u = SATURATION

        uni_torque_error = np.linspace(-limit_erro, limit_erro, 300)
        uni_speed_errorr = np.linspace(-limit_derro, limit_derro, 300)
        uni_output = np.linspace(-limit_delta_u, limit_delta_u, 300)

        erro = ctrl.Antecedent(uni_torque_error, 'torque_error')
        d_erro = ctrl.Antecedent(uni_speed_errorr, 'speed_error')
        saida = ctrl.Consequent(uni_output, 'voltage')
        
        s_e = 5.0
        erro['NB'] = fuzz.gaussmf(uni_torque_error, -limit_erro, s_e)
        erro['NS'] = fuzz.gaussmf(uni_torque_error, -limit_erro/2, s_e)
        erro['ZE'] = fuzz.gaussmf(uni_torque_error, 0, s_e * 0.1)
        erro['PS'] = fuzz.gaussmf(uni_torque_error, limit_erro/2, s_e)
        erro['PB'] = fuzz.gaussmf(uni_torque_error, limit_erro, s_e)
        
        s_d = 5.0
        d_erro['NB'] = fuzz.gaussmf(uni_speed_errorr, -limit_derro, s_d)
        d_erro['NS'] = fuzz.gaussmf(uni_speed_errorr, -limit_derro/2, s_d)
        d_erro['ZE'] = fuzz.gaussmf(uni_speed_errorr, 0, s_d)
        d_erro['PS'] = fuzz.gaussmf(uni_speed_errorr, limit_derro/2, s_d)
        d_erro['PB'] = fuzz.gaussmf(uni_speed_errorr, limit_derro, s_d)

        s_u = 5.0
        saida['NB'] = fuzz.gaussmf(uni_output, -limit_delta_u, s_u)
        saida['NS'] = fuzz.gaussmf(uni_output, -limit_delta_u/2, s_u)
        saida['ZE'] = fuzz.gaussmf(uni_output, 0, s_u * 0.1)
        saida['PS'] = fuzz.gaussmf(uni_output, limit_delta_u/2, s_u)
        saida['PB'] = fuzz.gaussmf(uni_output, limit_delta_u, s_u)


        rules = []
        rules.append(ctrl.Rule(erro['NB'] & d_erro['NB'], saida['NB']))
        rules.append(ctrl.Rule(erro['NB'] & d_erro['NS'], saida['NB']))
        rules.append(ctrl.Rule(erro['NB'] & d_erro['ZE'], saida['NB']))
        rules.append(ctrl.Rule(erro['NB'] & d_erro['PS'], saida['NS']))
        rules.append(ctrl.Rule(erro['NB'] & d_erro['PB'], saida['ZE']))

        rules.append(ctrl.Rule(erro['NS'] & d_erro['NB'], saida['NB']))
        rules.append(ctrl.Rule(erro['NS'] & d_erro['NS'], saida['NS']))
        rules.append(ctrl.Rule(erro['NS'] & d_erro['ZE'], saida['NS']))
        rules.append(ctrl.Rule(erro['NS'] & d_erro['PS'], saida['ZE']))
        rules.append(ctrl.Rule(erro['NS'] & d_erro['PB'], saida['PS']))

        rules.append(ctrl.Rule(erro['ZE'] & d_erro['NB'], saida['NS']))
        rules.append(ctrl.Rule(erro['ZE'] & d_erro['NS'], saida['NS']))
        rules.append(ctrl.Rule(erro['ZE'] & d_erro['ZE'], saida['ZE']))
        rules.append(ctrl.Rule(erro['ZE'] & d_erro['PS'], saida['PS']))
        rules.append(ctrl.Rule(erro['ZE'] & d_erro['PB'], saida['PS']))

        rules.append(ctrl.Rule(erro['PS'] & d_erro['NB'], saida['NS']))
        rules.append(ctrl.Rule(erro['PS'] & d_erro['NS'], saida['ZE']))
        rules.append(ctrl.Rule(erro['PS'] & d_erro['ZE'], saida['PS']))
        rules.append(ctrl.Rule(erro['PS'] & d_erro['PS'], saida['PS']))
        rules.append(ctrl.Rule(erro['PS'] & d_erro['PB'], saida['PB']))

        rules.append(ctrl.Rule(erro['PB'] & d_erro['NB'], saida['ZE']))
        rules.append(ctrl.Rule(erro['PB'] & d_erro['NS'], saida['PS']))
        rules.append(ctrl.Rule(erro['PB'] & d_erro['ZE'], saida['PB']))
        rules.append(ctrl.Rule(erro['PB'] & d_erro['PS'], saida['PB']))
        rules.append(ctrl.Rule(erro['PB'] & d_erro['PB'], saida['PB']))

        motor_ctrl = ctrl.ControlSystem(rules)
        cls._simulation_instance = ctrl.ControlSystemSimulation(motor_ctrl)

def PID_controller(tau, tau_ref, kp, ki, kd, dt, err_dict):
    error = tau_ref - tau
    
    if dt > 0:
        err_dict["integral"] += error * dt
        err_dict["integral"] = np.clip(err_dict["integral"], -SATURATION, SATURATION)
        derivative_error = (error - err_dict["prev_err"]) / dt
    else:
        derivative_error = 0

    out = error*(kp  + (ki * err_dict["integral"]) + (kd * derivative_error))
    err_dict["prev_err"] = error

    return np.clip(out, -SATURATION, SATURATION)

def Fuzzy_controller(tau, tau_ref, dt, err_dict):

    error = tau_ref - tau
    if dt > 0:
        d_error = (error - err_dict["prev_err"]) / dt
    else:
        d_error = 0

    fuzzy_sim = FuzzyFactory.get_instance()

    fuzzy_sim.input['torque_error'] = np.clip(error, -40, 40)
    fuzzy_sim.input['speed_error'] = np.clip(d_error, -70, 70)

    try:
        fuzzy_sim.compute()
        output = fuzzy_sim.output['voltage']
    except:
        output = 0

    prev_output = err_dict.get("prev_output", 0.0)
    ki = 20

    current_u = prev_output + output*ki
    current_u = np.clip(current_u, -SATURATION, SATURATION)

    err_dict["prev_err"] = error
    err_dict["prev_output"] = current_u

    return current_u

def Fuzzy_feeding_PID_controller(tau, tau_ref, kp, ki, kd, dt, err_dict):
    
    real_error = tau_ref - tau
    
    if dt > 0:
        real_d_error = (real_error - err_dict.get("prev_real_err", 0)) / dt
    else:
        real_d_error = 0

    
    fuzzy_sim = FuzzyFactory.get_instance()
    fuzzy_sim.input['torque_error'] = np.clip(real_error, -40, 40)
    fuzzy_sim.input['speed_error'] = np.clip(real_d_error, -70, 70)

    try:
        fuzzy_sim.compute()
        fuzzy_output = fuzzy_sim.output['voltage'] 
    except:
        fuzzy_output = 0

    
    pid_input = fuzzy_output 

    if dt > 0:
        err_dict["integral"] += pid_input * dt
        # Anti-windup
        err_dict["integral"] = np.clip(err_dict["integral"], -SATURATION, SATURATION)
        
        derivative_pid_input = (pid_input - err_dict.get("prev_fuzzy_out", 0)) / dt
    else:
        derivative_pid_input = 0

    out = pid_input*(kp  + (ki * err_dict["integral"]) + (kd * derivative_pid_input))
    out = np.clip(out, -SATURATION, SATURATION)
    # Atualização de estados
    err_dict["prev_real_err"] = real_error
    err_dict["prev_fuzzy_out"] = pid_input

    return np.clip(out, -SATURATION, SATURATION)