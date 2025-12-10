# config.py
import numpy as np

# Configurações de Simulação
TF = 8.0            # final time of simulation (seconds)
TS_MS = 0.01        # step time (seconds)
SATURATION = 42.0   # maximum control signal (voltage)

# Initial Estimated Motor Parameters
A_EST = 1.0
K_EST = 1.0