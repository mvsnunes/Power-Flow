# Newton-Raphson Power Flow Solver ⚡

This repository implements a Power Flow solver for electrical power systems using the **Newton-Raphson method**. It calculates bus voltages, power injections, and line losses in a power network.

## 📘 Overview

The Newton-Raphson method is widely used for solving the nonlinear algebraic equations that arise in power flow analysis due to its fast convergence properties.

This implementation:
- Solves the power flow using the Newton-Raphson method
- Handles PQ and PV bus types
- Reports:
  - Active and reactive power injections at all buses
  - Final bus voltages (magnitude and angle)
  - Active and reactive power losses per transmission line