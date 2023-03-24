from mpc_controller import mpc_controller


if __name__ == '__main__':
    mpcControl = mpc_controller()
    rel_dist = 30
    v0 = 20.0
    a0 = 2.0
    v_lead = 25.0
    v_max = 30.0
    safe_stop_dist = 15.0
    dt = 0.1
    N = 150
    mpcControl.mpc(rel_dist, v0, a0, v_lead, v_max, safe_stop_dist, dt, N)