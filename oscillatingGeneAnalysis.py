import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the parameters and initial conditions
gene0 = [0, 0, 0]
# Time points
t = np.linspace(0, 200, num=100)
# Initial production rates
p1, p2, p3 = 0.5, 0.5, 0.5
# Gammas
Gamma1, Gamma2, Gamma3 = 0.1, 0.5, 0.5
# n is the exponent in the Hill function
n, c = 9, 1
param = [p1, Gamma1, p2, Gamma2, p3, Gamma3, n, c]

# Define the simulator function
def simulator(variables, timepoint, param):
    Gene1, Gene2, Gene3 = variables
    p1, p2, p3 = param[0], param[2], param[4]
    Gamma1, Gamma2, Gamma3 = param[1], param[3], param[5]
    n, c = param[6], param[7]

    #Repression hill function
    dg1dt = (c ** n / (c ** n + Gene3)) * p1 - Gamma1 * Gene1
    #activation hill function
    dg2dt = (Gene1 ** n / (c ** n + Gene1)) * p2 - Gamma2 * Gene2
    dg3dt = (Gene2 ** n / (c ** n + Gene2)) * p3 - Gamma3 * Gene3

    return [dg1dt, dg2dt, dg3dt]

# Integrate the ODE system
y = odeint(simulator, gene0, t, args=(param,))

# Stability analysis
def jacobian_matrix(variables, timepoint, param):
    Gene1, Gene2, Gene3 = variables
    p1, p2, p3 = param[0], param[2], param[4]
    Gamma1, Gamma2, Gamma3 = param[1], param[3], param[5]
    n, c = param[6], param[7]

    # Calculate the elements of the Jacobian matrix
    df1dg1 = -Gamma1 - (c ** n * p1 * n * Gene3) / (c ** n + Gene3) ** 2
    df1dg3 = (c ** n * p1) / (c ** n + Gene3) - (c ** n * p1 * Gene1 * n) / (c ** n + Gene3) ** 2
    df2dg1 = (c ** n * p2 * n * Gene1) / (c ** n + Gene1) ** 2 - Gamma2
    df2dg2 = -(c ** n * p2 * n * Gene1 * Gene2) / (c ** n + Gene1) ** 2
    df3dg2 = (c ** n * p3 * n * Gene2) / (c ** n + Gene2) ** 2 - Gamma3
    df3dg3 = -(c ** n * p3 * n * Gene2 * Gene3) / (c ** n + Gene2) ** 2

    # Jacobian matrix
    jacobian = np.array([[df1dg1, 0, df1dg3],
                         [df2dg1, df2dg2, 0],
                         [0, df3dg2, df3dg3]])

    return jacobian

# Evaluate the Jacobian matrix at the steady state
steady_state = y[-1]
jacobian = jacobian_matrix(steady_state, 0, param)

# Compute the eigenvalues of the Jacobian matrix
eigenvalues, _ = np.linalg.eig(jacobian)

# Check stability
if all(eig_real < 0 for eig_real in eigenvalues.real):
    print("The gene regulatory network has a stable steady state.")
elif any(eig_real > 0 for eig_real in eigenvalues.real):
    print("The gene regulatory network has an unstable steady state.")
else:
    print("The stability of the steady state cannot be determined.")

# Plot the gene expression dynamics
plt.figure(figsize=(10, 6))
plt.plot(t, y[:, 0], label='Gene 1')
plt.plot(t, y[:, 1], label='Gene 2')
plt.plot(t, y[:, 2], label='Gene 3')
plt.xlabel('Time')
plt.ylabel('Gene Expression Level')
plt.title('Gene Regulatory Network Simulation')
plt.legend()
plt.grid(True)
plt.show()


# Fourier analysis for oscillation detection
fft_result = np.fft.fft(y, axis=0)
freqs = np.fft.fftfreq(len(t), t[1] - t[0])

# Plot Fourier spectrum to detect oscillations
plt.figure(figsize=(10, 6))
plt.plot(freqs, np.abs(fft_result[:, 0]), label='Gene 1')
plt.plot(freqs, np.abs(fft_result[:, 1]), label='Gene 2')
plt.plot(freqs, np.abs(fft_result[:, 2]), label='Gene 3')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Fourier Spectrum - (Oscillation Analysis)')
plt.legend()
plt.grid(True)
plt.show()


# Define perturbation function lol
def apply_perturbation(base_state, perturbation):
    return base_state + perturbation

# Define perturbation
perturbation = np.array([0.1, 0, 0])  # Ex: increase Gene1 expression by 0.1, I recommend playing around with the params

# Apply perturbation
y_perturbed = odeint(simulator, apply_perturbation(gene0, perturbation), t, args=(param,))

# Plot the dynamic response to perturbation
plt.figure(figsize=(10, 6))
plt.plot(t, y[:, 0], label='Gene 1 (No Perturbation)')
plt.plot(t, y_perturbed[:, 0], label='Gene 1 (With Perturbation)')
plt.xlabel('Time')
plt.ylabel('Gene Expression Level')
plt.title('Dynamic Response Analysis - Gene 1')
plt.legend()
plt.grid(True)
plt.show()