# main.py
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scy_int
import time

# Imports Modulares
from config import TF, TS_MS, A_EST, K_EST
from estimators import generate_data, qrd_rls_first_order
from optimizers import SimplexMethod, ParticleSwarmOptimization, GeneticAlgorithm
from simulation import setup_motor, run_simulation_cost, connected_systems_model, connected_systems_model_fuzzy, connected_systems_model_fuzzy_pid
from utils import get_ref
from controllers import FuzzyFactory

def plot_results(method_name, initial_pid, optimized_pid):
    print(f"\n Initial PID vs {method_name}...")
    t_eval = np.linspace(0, TF, 1000)
    
    d_init = {"integral": 0, "prev_err": 0, "prev_t": 0}
    res_init = scy_int.solve_ivp(connected_systems_model, [0, TF], [0, 0], 
                                 args=(get_ref, d_init, initial_pid), t_eval=t_eval, method='RK45')
    
    d_opt = {"integral": 0, "prev_err": 0, "prev_t": 0}
    res_opt = scy_int.solve_ivp(connected_systems_model, [0, TF], [0, 0], 
                                args=(get_ref, d_opt, optimized_pid), t_eval=t_eval, method='RK45')
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_eval, get_ref(t_eval, "step"), 'k--', label='Reference')
    plt.plot(t_eval, res_init.y[0], 'r:', label=f'Inicial {initial_pid}')
    plt.plot(t_eval, res_opt.y[0], 'b-', linewidth=2, label=f'{method_name} {np.round(optimized_pid, 2)}')
    plt.title(f"Optimization with {method_name}")
    plt.xlabel('Tempo (s)')
    plt.ylabel('Torque')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Control systems optimization')
    parser.add_argument('--simplex', action='store_true', help='Execute Simplex')
    parser.add_argument('--pso', action='store_true', help='Execute PSO')
    parser.add_argument('--ga', action='store_true', help='Execute Genetic Algorithm')
    parser.add_argument('--all', action='store_true', help='Execute all')
    parser.add_argument('--fuzzy', action='store_true', help='Execute fuzzy controller ')
    parser.add_argument('--fuzzy_simplex', action='store_true', help='Optimize Fuzzy-PID via Simplex')
    args = parser.parse_args()

    print("Model identification via QRD-RLS")
    y_data, u_data, _, _, _ = generate_data(1000, TS_MS)
    theta = qrd_rls_first_order(y_data, u_data, 0.99, 1000.0)
    
    a_estimated = -theta[-1, 0]
    k_estimated = theta[-1, 1]
    setup_motor(a_estimated, k_estimated)
    print(f"Estimated params: a={a_estimated:.4f}, k={k_estimated:.4f}")

    bounds = [[0, 200], [0, 20], [0, 10]] # Bound limits for Kp, Ki, Kd
    initial_guess = [10.0, 1.0, 0.5]

    if args.simplex or args.all:
        print("\nSimplex optimization")
        opt = SimplexMethod(run_simulation_cost, initial_guess, max_iter=30)
        best_pid = opt.optimize()
        plot_results("Simplex", initial_guess, best_pid)

    if args.pso or args.all:
        print("\nPSO optimization")
        opt = ParticleSwarmOptimization(run_simulation_cost, bounds, num_particles=15, max_iter=20)
        best_pid = opt.optimize()
        plot_results("PSO", initial_guess, best_pid)

    if args.ga or args.all:
        print("\nGenetic Algorithm optimization")
        opt = GeneticAlgorithm(run_simulation_cost, bounds, pop_size=15, generations=20)
        best_pid = opt.optimize()
        plot_results("Genetic Alg", initial_guess, best_pid)
    if args.fuzzy:
        print("\n Simulating Fuzzy Controller")
        t_eval = np.linspace(0, TF, 1000)
        d_fuzzy = {"integral": 0, "prev_err": 0, "prev_t": 0, "prev_output": 0}
        
        start_t = time.time()
        res_fuzzy = scy_int.solve_ivp(
            connected_systems_model_fuzzy, 
            [0, TF], 
            [0, 0], 
            args=(get_ref, d_fuzzy), 
            t_eval=t_eval, 
            method='RK45'
        )

        print(f"Fuzzy time: {time.time() - start_t:.2f}s")
        fuzz_control = FuzzyFactory.get_instance()
        f_c = fuzz_control.ctrl
        
        for i in f_c.antecedents:
            i.view()
        for j in f_c.consequents:
            j.view()

        plt.figure(figsize=(10, 6))
        plt.plot(t_eval, get_ref(t_eval, "step"), 'k--', label='Reference')
        plt.plot(t_eval, res_fuzzy.y[0], 'g-', linewidth=2, label='Fuzzy')
        plt.title("Fuzzy controller simulation")
        plt.xlabel('Tempo (s)'); plt.ylabel('Torque'); plt.grid(True); plt.legend()
        plt.show()
    if args.fuzzy_simplex:
        print("\n Optimizing Fuzzy-PID via Simplex ")
        
        initial_guess = [2.0, 20.0, 0.001] 
        opt = SimplexMethod(run_simulation_cost, initial_guess, max_iter=100)
        best_pid = opt.optimize()
        print(f"Best PID found: Kp={best_pid[0]:.4f}, Ki={best_pid[1]:.4f}, Kd={best_pid[2]:.4f}")

        print("[Plotting] Simulating Fuzzy-PID with optimized gains")
        t_eval = np.linspace(0, TF, 1000)
        d_opt = {"integral": 0, "prev_real_err": 0, "prev_fuzzy_out": 0, "prev_t": 0}
        
        res_opt = scy_int.solve_ivp(
            connected_systems_model_fuzzy_pid, 
            [0, TF], 
            [0, 0], 
            args=(get_ref, d_opt, best_pid), 
            t_eval=t_eval, 
            method='RK45'
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot(t_eval, get_ref(t_eval, "step"), 'k--', label='Reference')
        plt.plot(t_eval, res_opt.y[0], 'b-', linewidth=2, label=f'Fuzzy-PID')
        plt.title(f"Fuzzy-PID \nPID Gains: {np.round(best_pid, 3)}")
        plt.xlabel('Time (s)')
        plt.ylabel('Torque')
        plt.legend()
        plt.grid(True)
        plt.show()
    if not (args.simplex or args.pso or args.ga or args.all):
        print("\nNo method has been passed. Use --simplex, --pso, --ga or --all")

if __name__ == "__main__":
    main()