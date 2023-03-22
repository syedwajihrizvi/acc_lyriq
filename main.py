import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


def f(x, u, T, v_lead):
    return ca.horzcat(x[1], u, v_lead)


def predict_trajectory(x0, v0, x_lead0, v_lead, T, N, opti):
    X = opti.variable(N+1, 3)
    U = opti.variable(N, 1)
    J = opti.variable(N-1, 1)
    opti.subject_to(X[0, 0] == x0)
    opti.subject_to(X[0, 1] == v0)
    opti.subject_to(X[0, 2] == x_lead0)
    for i in range(N):
        opti.subject_to(U[i] >= -3)
        opti.subject_to(U[i] <= 3)
        opti.subject_to(X[i+1, 1] <= v_max)
        opti.subject_to(X[i+1, 1] >= 0)
        opti.subject_to(X[i+1, :] == X[i, :] + f(X[i, :], U[i], dt, v_lead)*dt)
        if (i > 0):
            opti.subject_to(J[i-1] <= 4)
            opti.subject_to(J[i-1] >= -4)
            opti.subject_to(J[i-1] == (U[i] - U[i-1])/dt)
    return X, U, J


# Set parameters
x0 = 0.0     # initial position
v0 = 20.0    # initial velocity
x_lead0 = 50.0  # lead car position
safeFollowingDistance = 15
v_lead = 20.0  # lead car velocity
v_max = 30.0  # maximum velocity
dt = 0.1     # discretization time
a0 = 2.0      # acceleration

# Set up optimization problem
opti = ca.Opti()

# Set up time horizon and number of control intervals
N = 150
T = opti.variable()
opti.subject_to(T >= 0.1)
opti.subject_to(T <= 10.0)

# Set up decision variables and constraints
X, U, J = predict_trajectory(x0, v0, x_lead0, v_lead, T, N, opti)

w1 = 1.0
w2 = 1.0
w3 = 20.0
opti.minimize(w1*ca.sumsqr(U) + w2*ca.sumsqr(J) +
              w3*ca.sumsqr(ca.fabs(ca.fabs(X[:, 0] - X[:, 2]) - safeFollowingDistance)))

opti.subject_to(X[:, 1] <= v_max)
opti.subject_to(X[:, 1] >= 0)

# Set up initial conditions and solve the problem
opti.set_initial(T, 1.0)
opti.solver('ipopt')
sol = opti.solve()

# Plot results
tgrid = np.linspace(0, (N+1)*dt, N+1)
plt.figure()
plt.plot(tgrid, sol.value(X)[:, 0]-sol.value(X)[:, 2], '-', label='position')
plt.xlabel('Time [s]')
plt.ylabel('Relative Distance [m]')
plt.title('Relative Distance between Ego and Lead Vehicle')

plt.figure()
plt.plot(tgrid, sol.value(X)[:, 1], '-', label='velocity')
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.title('Velocity of the Ego Vehicle')

plt.figure()
u = sol.value(U)
tgrid = np.linspace(0, (N)*dt, N)
plt.plot(tgrid, u, '-', label='acceleration')
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [m/s^2]')
plt.title('Acceleration of the Ego Vehicle')

plt.figure()
u = sol.value(U)
tgrid = np.linspace(0, (N-1)*dt, N-1)
plt.plot(tgrid, sol.value(J), '-', label='Jerk')
plt.xlabel('Time [s]')
plt.ylabel('Jerk [m/s^2]')
plt.title('Jerk of the Ego Vehicle')

plt.legend()
plt.show()
