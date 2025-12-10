# models.py
class DCMotor:
    def __init__(self, a, k, max_voltage):
        self.a = a
        self.k = k
        self.max_voltage = max_voltage

    def taup_by_model(self, tau, u):
        """ Calculate the derivative of torque based on the motor model."""
        return (self.k * u) - (self.a * self.k * tau)