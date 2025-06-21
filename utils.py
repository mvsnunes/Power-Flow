import numpy as np
import pandas as pd

def compute_admittance(z):
    """
    Calculate the admittance given an impedance.

    Parameters:
    z (float or complex): The impedance value. Must not be zero.

    Returns:
    float or complex: The admittance (1/z).

    Raises:
    ValueError: If the impedance is zero.
    """
    if z == 0:
        raise ValueError("The impedance cannot be zero.")
    return 1 / z

def compute_y_bus(impedance, shunt_admittance=None):
    """
    Build the bus admittance matrix (Y-bus) for a given network.

    Args:
        impedance (dict): A dictionary where keys are tuples representing node pairs (i, j)
            and values are the impedances (float or complex) between the nodes.
        shunt_admittance (dict, optional): A dictionary where keys are node pairs (i, j) and
            values are shunt admittances (float or complex). Defaults to None.

    Returns:
        numpy.ndarray: The bus admittance matrix (Y-bus) as a complex-valued 2D array.
    """
    if shunt_admittance is None:
        shunt_admittance = {}

    admittances = {par: compute_admittance(z) for par, z in impedance.items()}
    nodes = sorted(set(i for par in impedance for i in par))
    n = len(nodes)

    nodes_map = {no: idx for idx, no in enumerate(nodes)}

    Ybus = np.zeros((n, n), dtype=complex)

    for (i, j), y in admittances.items():
        idx_i, idx_j = nodes_map[i], nodes_map[j]
        Ybus[idx_i, idx_i] += y
        Ybus[idx_j, idx_j] += y
        Ybus[idx_i, idx_j] -= y
        Ybus[idx_j, idx_i] -= y

    for (i, j), y_sh in shunt_admittance.items():
        idx_i, idx_j = nodes_map[i], nodes_map[j]
        Ybus[idx_i, idx_i] += y_sh / 2
        Ybus[idx_j, idx_j] += y_sh / 2

    return Ybus


import numpy as np

def compute_jacobian(V, delta, Ybus, pv, pq):
    """
    Calcula a matriz Jacobiana para o fluxo de carga usando o método de Newton-Raphson.

    Args:
        V (numpy.ndarray): Módulos de tensão nas barras.
        delta (numpy.ndarray): Ângulos de tensão nas barras (em rad).
        Ybus (numpy.ndarray): Matriz de admitância nodal (complexa).
        pv (list of int): Índices das barras PV.
        pq (list of int): Índices das barras PQ.

    Returns:
        numpy.ndarray: Matriz Jacobiana.
    """
    N = len(V)
    npv = len(pv)
    npq = len(pq)
    ntheta = npv + npq
    nvar = ntheta + npq
    J = np.zeros((nvar, nvar))

    bus_delta = pv + pq
    bus_v = pq

    # J1: dP/dTheta
    for i, k in enumerate(bus_delta):
        for j, n in enumerate(bus_delta):
            if k == n:
                sum = 0
                for m in range(N):
                    if m != k:
                        Gkm = Ybus[k, m].real
                        Bkm = Ybus[k, m].imag
                        sum += V[m] * (Gkm * np.sin(delta[k] - delta[m]) - Bkm * np.cos(delta[k] - delta[m]))
                J[i, j] = -V[k] * sum
            else:
                Gkn = Ybus[k, n].real
                Bkn = Ybus[k, n].imag
                J[i, j] = V[k] * V[n] * (Gkn * np.sin(delta[k] - delta[n]) - Bkn * np.cos(delta[k] - delta[n]))

    # J2: dP/dV
    for i, k in enumerate(bus_delta):
        for j, n in enumerate(bus_v):
            Gkn = Ybus[k, n].real
            Bkn = Ybus[k, n].imag
            if k == n:
                sum = 0
                for m in range(N):
                    if m != k:
                        Gkm = Ybus[k, m].real
                        Bkm = Ybus[k, m].imag
                        sum += V[m] * (Gkm * np.cos(delta[k] - delta[m]) + Bkm * np.sin(delta[k] - delta[m]))
                J[i, ntheta + j] = V[k] * Ybus[k, k].real + sum
            else:
                J[i, ntheta + j] = V[k] * (Gkn * np.cos(delta[k] - delta[n]) + Bkn * np.sin(delta[k] - delta[n]))


    # J3: dQ/dTheta
    for i, k in enumerate(bus_v):
        for j, n in enumerate(bus_delta):
            Gkn = Ybus[k, n].real
            Bkn = Ybus[k, n].imag
            if k == n:
                sum = 0
                for m in range(N):
                    if m != k:
                        Gkm = Ybus[k, m].real
                        Bkm = Ybus[k, m].imag
                        sum += V[m] * (Gkm * np.cos(delta[k] - delta[m]) + Bkm * np.sin(delta[k] - delta[m]))
                J[ntheta + i, j] = V[k] * sum
            else:
                J[ntheta + i, j] = -V[k] * V[n] * (Gkn * np.cos(delta[k] - delta[n]) + Bkn * np.sin(delta[k] - delta[n]))

    # J4: dQ/dV
    for i, k in enumerate(bus_v):
        for j, n in enumerate(bus_v):
            Gkn = Ybus[k, n].real
            Bkn = Ybus[k, n].imag
            if k == n:
                sum = 0
                for m in range(N):
                    if m != k:
                        Gkm = Ybus[k, m].real
                        Bkm = Ybus[k, m].imag
                        sum += V[m] * (Gkm * np.sin(delta[k] - delta[m]) - Bkm * np.cos(delta[k] - delta[m]))
                J[ntheta + i, ntheta + j] = -V[k] * Ybus[k, k].imag - sum
            else:
                J[ntheta + i, ntheta + j] = -V[k] * (Gkn * np.sin(delta[k] - delta[n]) - Bkn * np.cos(delta[k] - delta[n]))

    return J



def compute_loss(V, theta, impedance, shunt_admittance, node_map):
    """
    Calculate the active and reactive power losses in each transmission line.

    Args:
        V (numpy.ndarray): Voltage magnitudes at all buses.
        theta (numpy.ndarray): Voltage angles at all buses (in radians).
        impedance (dict): Dictionary where keys are node pairs (i, k) and values are impedances (complex).
        shunt_admittance (dict): Dictionary where keys are node pairs (i, k) and values are shunt admittances (complex).
        node_map (dict): Mapping from bus identifiers to their corresponding indices in V and theta arrays.

    Returns:
        list of tuple: A list containing tuples of the form (i, k, active_loss, reactive_loss)
        where:
            - i (int): Sending bus.
            - k (int): Receiving bus.
            - active_loss (float): Active power loss (real part).
            - reactive_loss (float): Reactive power loss (imaginary part).
    """
    loss = []
    for (i, k), Z in impedance.items():
        idx_i = node_map[i]
        idx_k = node_map[k]

        Vi = V[idx_i] * np.exp(1j * theta[idx_i])
        Vk = V[idx_k] * np.exp(1j * theta[idx_k])

        Y_sh = shunt_admittance.get((i, k), 0)  

        Iik = (Vi - Vk) / Z
        Iki = (Vk - Vi) / Z

        Sik = Vi * np.conj(Iik)
        Ski = Vk * np.conj(Iki)
        
        Ish_i = Vi * (Y_sh / 2)
        Ish_k = Vk * (Y_sh / 2)
        Ssh_i = Vi * np.conj(Ish_i)
        Ssh_k = Vk * np.conj(Ish_k)

        Sperda = Sik + Ski + Ssh_i + Ssh_k

        loss.append((i, k, Sperda.real, Sperda.imag))

    return loss




def compute_power(V, theta, Ybus):
    """
    Calculate the active and reactive power injected at each bus.

    Args:
        V (numpy.ndarray): Voltage magnitudes at all buses.
        theta (numpy.ndarray): Voltage angles at all buses (in radians).
        Ybus (numpy.ndarray): Bus admittance matrix (complex-valued).

    Returns:
        tuple:
            - P (numpy.ndarray): Active power injected at each bus.
            - Q (numpy.ndarray): Reactive power injected at each bus.
    """
    n = len(V)
    P = np.zeros(n)
    Q = np.zeros(n)
    for i in range(n):
        for k in range(n):
            P[i] += V[i] * V[k] * (
                Ybus[i, k].real * np.cos(theta[i] - theta[k]) +
                Ybus[i, k].imag * np.sin(theta[i] - theta[k])
            )
            Q[i] += V[i] * V[k] * (
                Ybus[i, k].real * np.sin(theta[i] - theta[k]) -
                Ybus[i, k].imag * np.cos(theta[i] - theta[k])
            )
    return P, Q

def newton_power_flow(Ybus, bus_types, P_spec, Q_spec, V_spec,
                      baseMVA=100, tol_MVA=0.1, max_iter=20):
    """
    Solves the power flow using Newton-Raphson with MVA-based convergence and step size control.
    Parameters:
        Ybus (ndarray): N x N bus admittance matrix (complex).
        bus_types (list): List of bus types (0=Slack, 1=PV, 2=PQ).
        P_spec (array): Specified active powers (pu).
        Q_spec (array): Specified reactive powers for PQ buses (pu).
        V_spec (array): Initial voltage magnitudes (pu).
        baseMVA (float): Base power in MVA.
        tol_MVA (float): Convergence tolerance in MVA.
        max_iter (int): Maximum number of iterations.
    Returns:
        V_full (array): Final voltage magnitudes (pu).
        theta_full (array): Final voltage angles (rad).
        P_calc (array): Final calculated active power injections (pu).
        Q_calc (array): Final calculated reactive power injections (pu).
    """
    nb = len(bus_types)
    slack_idx = bus_types.index(0)
    PV_indices = [i for i,t in enumerate(bus_types) if t==1]
    PQ_indices = [i for i,t in enumerate(bus_types) if t==2]
    V = np.array(V_spec, dtype=float)
    theta = np.zeros(nb)
    theta[slack_idx] = 0.0
    ang_idx = [i for i in range(nb) if bus_types[i] != 0]
    V_idx = PQ_indices.copy()
    x = np.concatenate((theta[ang_idx], V[V_idx]))

    def evaluate_state(x):
        theta_full = theta.copy()
        for k,i in enumerate(ang_idx):
            theta_full[i] = x[k]
        V_full = V.copy()
        for k,i in enumerate(V_idx):
            V_full[i] = x[len(ang_idx) + k]
        P_calc = np.zeros(nb)
        Q_calc = np.zeros(nb)
        for i in range(nb):
            for j in range(nb):
                G = Ybus[i,j].real
                B = Ybus[i,j].imag
                dth = theta_full[i] - theta_full[j]
                P_calc[i] += V_full[i]*V_full[j]*(G*np.cos(dth) + B*np.sin(dth))
                Q_calc[i] += V_full[i]*V_full[j]*(G*np.sin(dth) - B*np.cos(dth))
        return theta_full, V_full, P_calc, Q_calc

    def mismatches(x):
        theta_full, V_full, P_calc, Q_calc = evaluate_state(x)
        F = []
        for i in range(nb):
            if bus_types[i] == 1:
                F.append(P_calc[i] - P_spec[i])
            elif bus_types[i] == 2:
                F.append(P_calc[i] - P_spec[i])
        for i in range(nb):
            if bus_types[i] == 2:
                F.append(Q_calc[i] - Q_spec[i])
        return np.array(F)

    for it in range(1, max_iter+1):
        F = mismatches(x)
        max_mismatch = np.max(np.abs(F) * baseMVA)
        print(f"Iter {it}: max mismatch = {max_mismatch:.6f} MVA")
        if max_mismatch < tol_MVA:
            print("Converged within MVA tolerance.")
            break
        n = len(x)
        J = np.zeros((len(F), n))
        eps = 1e-6
        F0 = F.copy()
        for j in range(n):
            x_pert = x.copy()
            x_pert[j] += eps
            F_pert = mismatches(x_pert)
            J[:,j] = (F_pert - F0) / eps
        try:
            dx = np.linalg.solve(J, -F0)
        except np.linalg.LinAlgError:
            print("Jacobian is singular. Stopping.")
            break
        alpha = 1.0
        success = False
        f0_norm = np.max(np.abs(F0))
        while alpha > 1e-4:
            x_new = x + alpha * dx
            F_new = mismatches(x_new)
            fnew_norm = np.max(np.abs(F_new))
            if fnew_norm < f0_norm:
                x = x_new
                success = True
                break
            alpha *= 0.5
        if not success:
            x += alpha * dx
            print("Reduced step without improvement; continuing...")

    else:
        print("Did not converge within iteration limit.")

    theta_full, V_full, P_calc, Q_calc = evaluate_state(x)
    return V_full, theta_full, P_calc, Q_calc

