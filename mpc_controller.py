import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

class mpc_controller:
    def f(x, u, v_lead):
        return ca.horzcat((v_lead - x), u)


    def predict_trajectory(rel_dist, v0, a0, v_lead, v_max, dt, N, opti):
        X = opti.variable(N+1, 4)
        U = opti.variable(N, 1)
        J = opti.variable(N-1, 1)
        opti.subject_to(X[0, 0] == rel_dist)
        opti.subject_to(X[0, 1] == v0)
        
        for i in range(N):
            opti.subject_to(U[i] >= -3)
            opti.subject_to(U[i] <= 3)
            opti.subject_to(X[i+1, 1] <= v_max)
            opti.subject_to(X[i+1, 1] >= 0)
            opti.subject_to(X[i+1, :] == X[i, :] + mpc_controller.f(X[i, 1], U[i], v_lead)*dt)

            if i < N-1:
                opti.subject_to(J[i] <= 4)
                opti.subject_to(J[i] >= -4)
                if i ==0:
                    opti.subject_to(J[i] == (U[i] - a0)/dt)
                else:
                    opti.subject_to(J[i] == (U[i] - U[i-1])/dt)
        return X, U, J



    def mpc(self,rel_dist,v0,a0,v_lead,v_max, safe_stop_dist,dt, N):

        # Set up optimization problem
        opti = ca.Opti()

        # Set up time horizon and number of control intervals
        T = opti.variable()
        opti.subject_to(T >= 0.1)
        opti.subject_to(T <= 10.0)

        # Set up decision variables and constraints
        X, U, J = mpc_controller.predict_trajectory(rel_dist, v0, a0, v_lead, v_max, dt, N, opti)

        w1 = 20.0
        w2 = 20.0
        w3 = 10.0
        opti.minimize(w1*ca.sumsqr(U) + w2*ca.sumsqr(J) +
                    w3*ca.sumsqr(ca.fabs(X[:,0] - safe_stop_dist)))

        opti.subject_to(X[:, 1] <= v_max)
        opti.subject_to(X[:, 1] >= 0)

        # Set up initial conditions and solve the problem
        opti.set_initial(T, 1.0)
        opti.solver('ipopt')
        sol = opti.solve()

        # # Plot results
        # tgrid = np.linspace(0, (N+1)*dt, N+1)
        # plt.figure()
        # plt.plot(tgrid, sol.value(X)[:, 0], '-', label='position')
        # plt.xlabel('Time [s]')
        # plt.ylabel('Relative Distance [m]')
        # plt.title('Relative Distance between Ego and Lead Vehicle')

        # plt.figure()
        # plt.plot(tgrid, sol.value(X)[:, 1], '-', label='velocity')
        # plt.xlabel('Time [s]')
        # plt.ylabel('Velocity [m/s]')
        # plt.title('Velocity of the Ego Vehicle')

        # plt.figure()
        # u = sol.value(U)
        # tgrid = np.linspace(0, (N)*dt, N)
        # plt.plot(tgrid, u, '-', label='acceleration')
        # plt.xlabel('Time [s]')
        # plt.ylabel('Acceleration [m/s^2]')
        # plt.title('Acceleration of the Ego Vehicle')

        # plt.figure()
        # tgrid = np.linspace(0, (N-1)*dt, N-1)
        # plt.plot(tgrid, sol.value(J), '-', label='Jerk')
        # plt.xlabel('Time [s]')
        # plt.ylabel('Jerk [m/s^2]')
        # plt.title('Jerk of the Ego Vehicle')

        # plt.legend()
        # plt.show()
