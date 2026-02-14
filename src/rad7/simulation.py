"""
Simulation module for Rad7 data.
Solves the diffusion PDE numerically.
Refactored to avoid global variables.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Dict, Any, List

class DiffusionModel:
    """
    Class to handle the diffusion simulation parameters and solving.
    """
    def __init__(self, 
                 membrane_thickness_m: float = 2e-5, # Default guess
                 membrane_area_m2: float = 1e-4,     # Default guess
                 vol_cold_m3: float = 0.0008,        # Rad7 volume?
                 lambda_rn222: float = 2.1e-6,       # Decay constant (approx)
                 c_hot: float = 1000.0):             # Hot side concentration
        
        self.L = membrane_thickness_m
        self.A = membrane_area_m2
        self.V = vol_cold_m3
        self.lam = lambda_rn222
        self.C_H = c_hot
        
        # Grid parameters
        self.N = 30 # spatial steps
        self.dx = self.L / self.N
        self.x_grid = np.linspace(0, self.L, self.N)
        
        # State
        self.solm: Any = None # Membrane solution
        
    def _pde_membrane(self, t, C, D):
        """Diffusive transport in membrane."""
        dCdt = np.zeros(self.N)
        
        # BC at x=0 (Hot Side): Dirichlet C(0,t) = C_H * exp(-lam * t)
        # Assuming C_H decays? Original code: vs.C_H * np.exp(-vs.lambda_ * t)
        C_boundary = self.C_H * np.exp(-self.lam * t)
        C[0] = C_boundary
        
        # Interior points
        # D * d2C/dx2 - lam * C
        # Finite difference: (C[i+1] - 2C[i] + C[i-1]) / dx^2
        for i in range(1, self.N - 1):
            dCdt[i] = D * (C[i+1] - 2*C[i] + C[i-1]) / self.dx**2 - self.lam * C[i]
            
        # BC at x=L (Cold Side): Neumann?
        # Original: dCdt[-1] = - D * (C[-1] - C[-3]) / (2*dx) - vs.lambda_ * C[-1]
        # This looks like a flux logic but applied to the concentration at the boundary?
        # Actually in original code:
        # pde_InMembrane returns dCdt.
        # It sets dCdt[-1] using one-sided derivative?
        
        # Let's trust original discretization for now
        dCdt[-1] = - D * (C[-1] - C[-3]) / (2*self.dx) - self.lam * C[-1]
        
        return dCdt

    def _pde_cold_side(self, t, C_cold, P, D):
        """
        Concentration evolution in the cold volume.
        dC/dt = (A/V) * Flux - lam * C
        Flux J = -P * dC/dx | x=L
        Original code used 'P' in pde_coldSide:
        - P * (A/V) * (solm.sol(t)[-1] - solm.sol(t)[-3]) / (2*dx) ...
        Wait, 'P' usually means Permeability, but here it acts like a scaling factor on flux?
        Or maybe P is Porosity?
        
        Actually, looking at `model_coldSide`:
        dCdt = - P * (Area/Vol) * (Grad_C_membrane) - Decay
        
        We need the gradient at x=L from the membrane solution.
        The membrane solution is solved FIRST in the original code, then interpolated.
        """
        
        # Get membrane concentration at x=L (index -1) and x=L-dx (index -2) etc.
        # We need to query the dense output of the membrane solver
        C_mem_profile = self.solm.sol(t)
        
        # Gradient approx at Boundary L
        grad_C = (C_mem_profile[-1] - C_mem_profile[-3]) / (2 * self.dx)
        
        # Flux into cold side
        # J = -D * grad_C? 
        # Original code used `P` here. Let's stick to their parameter naming `P`.
        # NOTE: D was used inside membrane PDE. P is used for transfer to cold side?
        # Maybe P represents the partition coefficient * Area / Volume?
        
        dCdt = - P * (self.A / self.V) * grad_C - self.lam * C_cold
        
        return dCdt

    def solve(self, times: np.ndarray, D: float, P: float, tau_cold: float) -> np.ndarray:
        """
        Solves the system for given parameters.
        Returns concentration array matching `times`.
        """
        self.lam = 1.0 / (tau_cold * 86400.0) # Convert days to Hz? Original: 1/tau_cold/86400
        
        # 1. Solve Membrane PDE
        C0_mem = np.zeros(self.N)
        t_span = [times[0], times[-1]]
        
        self.solm = solve_ivp(
            fun=lambda t, y: self._pde_membrane(t, y, D),
            t_span=t_span,
            y0=C0_mem,
            t_eval=times,
            method='RK45',
            dense_output=True
        )
        
        # 2. Solve Cold Side ODE
        # Initial concentration 0? Original: vs.C_initial
        C0_cold = [0.0] 
        
        sol_cold = solve_ivp(
            fun=lambda t, y: self._pde_cold_side(t, y, P, D),
            t_span=t_span,
            y0=C0_cold,
            t_eval=times,
            method='RK45',
            dense_output=True
        )
        
        return sol_cold.y[0]

def solve_diffusion_model(time_sec: np.ndarray, 
                         D: float, 
                         P: float, 
                         tau_cold_days: float,
                         config: Dict[str, float]) -> np.ndarray:
    """
    Wrapper for functional usage.
    """
    model = DiffusionModel(
        membrane_thickness_m=config.get('membrane_thickness', 2e-5),
        membrane_area_m2=config.get('membrane_area', 1e-4),
        vol_cold_m3=config.get('volume_cold', 0.0008),
        c_hot=config.get('c_hot', 1000.0)
    )
    
    return model.solve(time_sec, D, P, tau_cold_days)
