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