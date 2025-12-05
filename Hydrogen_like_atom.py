"""
HYDROGEN-LIKE ATOM ORBITAL VISUALIZATION TOOL
Complete visualization and analysis of hydrogen-like atomic orbitals
with quantum corrections and advanced features.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.special import genlaguerre as scipy_genlaguerre
from scipy.special import eval_genlaguerre
from scipy.special import lpmv as scipy_lpmv
from scipy.optimize import brentq
from math import factorial, sqrt, pi
import time
import traceback
import scipy.constants as const
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    print("Warning: 'tqdm' library not found. Progress bars will be disabled.")
    print("Install it with: pip install tqdm")
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get('total', 0)
            self.iterable = args[0] if args else None
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def __iter__(self):
            return iter(self.iterable) if self.iterable else iter([])
        def __next__(self):
            return next(self.iterable)
        def update(self, n=1):
            pass
        def set_postfix_str(self, s):
            pass
    TQDM_AVAILABLE = False

a0 = 1.0                               
HARTREE_TO_EV = 27.211386245988                     
FINE_STRUCTURE_CONSTANT = 1/137.036                                    

m_e_c2_eV = (const.m_e * const.c**2) / const.e                                   
m_p_eV = (const.m_p * const.c**2) / const.e                                    
g_p = 5.5856946893                   
I_proton = 0.5               
s_electron = 0.5                

ORBITAL_LABELS = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g', 5: 'h', 6: 'i', 7: 'j'}

xp = np
try:
    import cupy as cp
    print("CuPy found. GPU acceleration is available.")
except ImportError:
    cp = None
    print("CuPy not found. Using NumPy for all calculations.")

laguerre_kernel_code = r'''
extern "C" __global__
void laguerre_poly(const double* rho, double* out, int n_poly, int k_poly, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    double x = rho[tid];
    if (n_poly == 0) {
        out[tid] = 1.0;
        return;
    }
    double L_prev = 1.0;
    double L_curr = 1.0 + k_poly - x;
    if (n_poly == 1) {
        out[tid] = L_curr;
        return;
    }
    for (int i = 1; i < n_poly; ++i) {
        double L_next = ((2.0 * i + 1.0 + k_poly - x) * L_curr - (i + k_poly) * L_prev) / (i + 1.0);
        L_prev = L_curr;
        L_curr = L_next;
    }
    out[tid] = L_curr;
}
'''
laguerre_kernel = None
if cp:
    try:
        laguerre_kernel = cp.RawKernel(laguerre_kernel_code, 'laguerre_poly')
    except Exception as e:
        print(f"Warning: Could not compile CUDA kernel: {e}")
        laguerre_kernel = None

def set_backend(use_gpu=False):
    """Set computational backend to GPU (CuPy) or CPU (NumPy)"""
    global xp
    if use_gpu and cp:
        xp = cp
        print("Backend set to CuPy (GPU). CUDA kernels will be used.")
    else:
        xp = np
        print("Backend set to NumPy (CPU).")

class CoordinateCache:
    """Cache coordinate transformations to avoid redundant calculations"""
    def __init__(self):
        self.cache = {}
    
    def get_spherical(self, X, Y, Z, grid_id):
        """Get or compute spherical coordinates"""
        if grid_id not in self.cache:
            R = xp.sqrt(X**2 + Y**2 + Z**2)
            R_safe = xp.where(R == 0, 1e-12, R)
            Theta = xp.arccos(xp.clip(Z / R_safe, -1.0, 1.0))
            Phi = xp.arctan2(Y, X)
            self.cache[grid_id] = (R, Theta, Phi)
        return self.cache[grid_id]
    
    def clear(self):
        """Clear cache to free memory"""
        self.cache = {}

coord_cache = CoordinateCache()

def get_laguerre(n, k, x):
    """
    Compute generalized Laguerre polynomial L_n^k(x)
    Uses GPU kernel if available, otherwise falls back to scipy
    """
    if xp == cp and laguerre_kernel is not None:
        x = xp.asarray(x) 
        x_float64 = x.astype(cp.float64)
        L = cp.empty_like(x_float64, dtype=cp.float64)
        block_size = 256
        grid_size = (x.size + block_size - 1) // block_size
        try:
            laguerre_kernel((grid_size,), (block_size,), (x_float64, L, n, k, x.size))
            return L
        except Exception as e:
            print(f"Warning: GPU kernel failed, falling back to CPU: {e}")

    x_cpu = cp.asnumpy(x) if hasattr(x, 'get') else x
                                                        
    result_cpu = eval_genlaguerre(n, k, x_cpu).astype(np.float64)
    return xp.asarray(result_cpu)

def R_nl(r, n, l, Z):
    """
    Radial wavefunction for hydrogen-like atom
    
    Parameters:
    -----------
    r : array_like
        Radial distance
    n : int
        Principal quantum number (n ≥ 1)
    l : int
        Angular momentum quantum number (0 ≤ l < n)
    Z : int
        Atomic number
    
    Returns:
    --------
    R : array_like
        Radial wavefunction values
    """
    r = xp.asarray(r)
    rho = (2.0 * Z * r / (n * a0)).astype(xp.float64)

    if n - l - 1 < 0:
        raise ValueError(f"Invalid quantum numbers: n={n}, l={l}. Must have n > l.")

    try:
        fact_n_min_l = factorial(n - l - 1)
        fact_n_plus_l = factorial(n + l)
        N_factor = ((2.0 * Z / (n * a0))**3 * fact_n_min_l) / (2 * n * fact_n_plus_l)
        N = xp.sqrt(N_factor)
    except (ValueError, OverflowError):
        from scipy.special import loggamma
        log_fact_n_min_l = loggamma(n - l)
        log_fact_n_plus_l = loggamma(n + l + 1)
        log_N_factor = 3 * xp.log(2.0 * Z / (n * a0)) + log_fact_n_min_l - xp.log(2 * n) - log_fact_n_plus_l
        N = xp.exp(log_N_factor / 2.0)

    L = get_laguerre(n - l - 1, 2 * l + 1, rho)

    result = N * xp.exp(-rho / 2.0) * (rho**l) * L
    
    return result

def Y_lm(l, m, theta, phi, real=False):
    """
    Spherical harmonics Y_l^m(θ, φ)
    
    Parameters:
    -----------
    l : int
        Angular momentum quantum number
    m : int
        Magnetic quantum number (-l ≤ m ≤ l)
    theta : array_like
        Polar angle [0, π]
    phi : array_like
        Azimuthal angle [0, 2π]
    real : bool
        If True, return real spherical harmonics
    
    Returns:
    --------
    Y : array_like (complex or real)
        Spherical harmonic values
    """
    if abs(m) > l:
        return xp.zeros_like(theta, dtype=xp.complex128 if not real else xp.float64)
    
    theta = xp.asarray(theta)
    phi = xp.asarray(phi)
    
    m_abs = abs(m)

    try:
        cos_theta = xp.cos(theta)
        legendre_poly_vals = scipy_lpmv(m_abs, l, cos_theta)
    except TypeError:
                                                                          
        print("Warning: SciPy-CuPy dispatch failed. Falling back to CPU for lpmv.")
        cos_theta_cpu = cp.asnumpy(xp.cos(theta)) if hasattr(xp, 'get') else xp.cos(theta)
        legendre_poly_vals_cpu = scipy_lpmv(m_abs, l, cos_theta_cpu)
        legendre_poly_vals = xp.asarray(legendre_poly_vals_cpu)

    legendre_poly = xp.asarray(legendre_poly_vals, dtype=xp.float64)

    try:
        norm_factor = ((2 * l + 1) / (4 * pi)) * (factorial(l - m_abs) / factorial(l + m_abs))
        norm = xp.sqrt(norm_factor)
    except (ValueError, OverflowError):
        from scipy.special import loggamma
        log_fact_l_min_m = loggamma(l - m_abs + 1)
        log_fact_l_plus_m = loggamma(l + m_abs + 1)
        norm = xp.exp(0.5 * (xp.log((2 * l + 1) / (4 * pi)) + log_fact_l_min_m - log_fact_l_plus_m))

    azimuthal_part = xp.exp(1j * m * phi.astype(xp.complex128))

    Y = norm * legendre_poly * azimuthal_part

    if m < 0:
        Y = ((-1)**m) * xp.conj(Y)
    
    if not real:
        return Y.astype(xp.complex128)

    if m == 0:
        return Y.real.astype(xp.float64)

    if m > 0:
                                                     
        val = sqrt(2.0) * ((-1)**m) * Y.real
    else:        

        azimuthal_part_pos = xp.exp(1j * m_abs * phi.astype(xp.complex128))
        Y_pos_m = norm * legendre_poly * azimuthal_part_pos                           
        val = sqrt(2.0) * ((-1)**m) * Y_pos_m.imag
    
    return val.astype(xp.float64)

def psi_nlm(r, theta, phi, n, l, m, Z, real=False):
    """
    Complete hydrogen-like atom wavefunction Ψ_nlm(r, θ, φ)
    
    Returns:
    --------
    psi : array_like
        Complete wavefunction
    """
    radial_part = R_nl(r, n, l, Z)
    angular_part = Y_lm(l, m, theta, phi, real=real)
    
    return radial_part * angular_part

def calculate_energy(n, Z):
    """Calculate energy eigenvalue in Hartree and eV"""
    E_hartree = -Z**2 / (2 * n**2)
    E_eV = E_hartree * HARTREE_TO_EV
    return E_hartree, E_eV

def calculate_fine_structure(n, l, j, Z):
    """
    Calculate fine structure correction (relativistic + spin-orbit)
    
    Using the standard formula:
    ΔE_fs = (E_n * (Zα)² / n²) * [n/(j+1/2) - 3/4]
    
    This combines relativistic and spin-orbit corrections.
    """
    if l == 0:
        j = 0.5                           
    
    E_n = -Z**2 / (2 * n**2)              
    alpha = FINE_STRUCTURE_CONSTANT
    
    correction = E_n * (Z * alpha)**2 / n**2 * (n / (j + 0.5) - 0.75)
    return correction * HARTREE_TO_EV

def relativistic_correction(n, l, Z):
    """
    Calculate relativistic (mass-velocity) correction in eV
    
    Formula: ΔE_rel = -(Zα)⁴ * m_e*c² / (2n³) * [n/(l+1/2) - 3/4]
    
    CORRECTED: Changed denominator from n⁴ to n³
    """
    alpha = FINE_STRUCTURE_CONSTANT
    
    term1 = -(Z * alpha)**4 * m_e_c2_eV / (2.0 * n**3)
    term2 = n / (l + 0.5) - 0.75
    
    return term1 * term2

def darwin_correction(n, l, Z):
    """
    Calculate Darwin term correction in eV
    Only non-zero for l=0 (s-orbitals)
    
    Formula: ΔE_darwin = (Zα)⁴ * m_e*c² / (2n³)
    """
    if l != 0:
        return 0.0
        
    alpha = FINE_STRUCTURE_CONSTANT
    delta_E_eV = (Z * alpha)**4 * m_e_c2_eV / (2.0 * n**3)
    return delta_E_eV

def spin_orbit_correction(n, l, j, Z):
    """
    Calculate spin-orbit coupling correction in eV
    Only non-zero for l > 0
    
    Formula: ΔE_SO = (Zα)⁴ * m_e*c² / (4n³) * [j(j+1) - l(l+1) - s(s+1)] / [l(l+1/2)(l+1)]
    
    CORRECTED: Changed denominator from n⁴ to n³
    """
    if l == 0:
        return 0.0

    alpha = FINE_STRUCTURE_CONSTANT

    spin_term_numerator = j*(j+1) - l*(l+1) - s_electron*(s_electron+1)

    spin_term_denominator = l * (l + 0.5) * (l + 1)
    
    if spin_term_denominator == 0:
        return 0.0
    
    term1 = (Z * alpha)**4 * m_e_c2_eV / (4.0 * n**3)
    
    return term1 * spin_term_numerator / spin_term_denominator

def hyperfine_splitting(n, l, j, F, Z):
    """
    Calculate hyperfine splitting correction in eV (Fermi contact term)
    Only significant for l=0 (s-orbitals) and Z=1 (hydrogen)
    
    Formula: ΔE_hfs = (4/3) * (Zα)⁴ * (g_p * m_e / m_p) * (m_e*c²) / n³ 
                      * [F(F+1) - I(I+1) - j(j+1)]
    
    CORRECTED: Changed coefficient from 2/3 to 4/3 and Z³ to Z⁴
    """
    if l != 0:
        return 0.0
    if Z != 1:
                                               
        return 0.0

    alpha = FINE_STRUCTURE_CONSTANT

    term1 = (4.0/3.0) * (Z * alpha)**4
    term2 = g_p * (const.m_e / const.m_p)
    term3 = m_e_c2_eV / (n**3)
    
    spin_term = F*(F+1) - I_proton*(I_proton+1) - j*(j+1)

    return term1 * term2 * term3 * spin_term

def find_radial_nodes(n, l, Z, r_max=None):
    """
    Find radial nodes by finding the roots of the Laguerre polynomial
    
    Returns:
    --------
    nodes : array
        Positions of radial nodes in units of a₀
    """
    n_nodes = n - l - 1
    if n_nodes == 0:
        return np.array([])
    
    try:
                                                                       
        poly = scipy_genlaguerre(n - l - 1, 2 * l + 1)

        rho_nodes = poly.roots

        r_nodes = rho_nodes * (n * a0) / (2.0 * Z)

        return np.sort(r_nodes[r_nodes > 0])                         
        
    except Exception as e:
        print(f"Warning: Could not find nodes using polynomial roots: {e}")
        return np.array([])

def check_normalization(n, l, m, Z, n_samples=100000):
    """
    Verify wavefunction normalization using Monte Carlo integration
    
    Returns:
    --------
    integral : float
        Value of normalization integral (should be ≈ 1.0)
    """
    r_max = n * (n + 20) / Z
    
    print(f"Checking normalization with {n_samples} Monte Carlo samples...")

    x = np.random.uniform(-r_max, r_max, n_samples)
    y = np.random.uniform(-r_max, r_max, n_samples)
    z = np.random.uniform(-r_max, r_max, n_samples)
    
    R = np.sqrt(x**2 + y**2 + z**2)
    R_safe = np.where(R == 0, 1e-12, R)
    Theta = np.arccos(np.clip(z / R_safe, -1.0, 1.0))
    Phi = np.arctan2(y, x)
    
    psi_mc = psi_nlm(R, Theta, Phi, n, l, m, Z, real=True)
    if hasattr(psi_mc, 'get'):
        psi_mc = psi_mc.get()
        
    prob_density_mc = np.abs(psi_mc)**2
    
    volume = (2 * r_max)**3
    integral = volume * np.mean(prob_density_mc)
    
    return integral

def calculate_expectation_values(n, l, Z):
    """
    Calculate expectation values <r>, <r²>, and <1/r>
    
    Returns analytical values using standard formulas
    """
                                       
    r_mean = (a0 / (2*Z)) * (3*n**2 - l*(l+1))
    r2_mean = (a0**2 * n**2 / (2*Z**2)) * (5*n**2 + 1 - 3*l*(l+1))
    r_inv_mean = Z / (a0 * n**2)
    
    return {
        '<r>': r_mean,
        '<r²>': r2_mean,
        '<1/r>': r_inv_mean
    }

def print_orbital_info(n, l, m, Z, show_fine_structure=False):
    """Display comprehensive orbital properties"""
    E_hartree, E_eV = calculate_energy(n, Z)
    n_radial_nodes = n - l - 1
    n_angular_nodes = l
    
    orbital_label = f"{n}{ORBITAL_LABELS.get(l, '?')}"
    
    print(f"\n{'='*70}")
    print(f"  HYDROGEN-LIKE ATOM ORBITAL: {orbital_label}")
    print(f"{'='*70}")
    print(f"  Ion: Z = {Z} (e.g., {'H' if Z==1 else 'He+' if Z==2 else f'Li^{Z-1}+'})")
    print(f"  Quantum Numbers: n = {n}, l = {l}, m = {m}")
    print(f"  Orbital Designation: {orbital_label}")
    print(f"{'-'*70}")
    print(f"  Energy (Bohr model):")
    print(f"    E_{n} = {E_hartree:.8f} Hartree")
    print(f"    E_{n} = {E_eV:.6f} eV")
    print(f"{'-'*70}")
    
    if show_fine_structure:
        if l == 0:
            j = 0.5
            E_fs = calculate_fine_structure(n, l, j, Z)
            print(f"  Fine Structure Correction (j = 1/2):")
            print(f"    ΔE = {E_fs:.6e} eV")
        else:
            j_minus = l - 0.5
            j_plus = l + 0.5
            E_fs_minus = calculate_fine_structure(n, l, j_minus, Z)
            E_fs_plus = calculate_fine_structure(n, l, j_plus, Z)
            print(f"  Fine Structure Corrections:")
            print(f"    j = {l} - 1/2 = {j_minus}: ΔE = {E_fs_minus:.6e} eV")
            print(f"    j = {l} + 1/2 = {j_plus}: ΔE = {E_fs_plus:.6e} eV")
            print(f"    Splitting: {abs(E_fs_minus - E_fs_plus):.6e} eV")
        print(f"{'-'*70}")
    
    print(f"  Node Structure:")
    print(f"    Radial nodes: {n_radial_nodes}")
    print(f"    Angular nodes: {n_angular_nodes}")
    print(f"    Total nodes: {n_radial_nodes + n_angular_nodes}")
    
    if n_radial_nodes > 0:
        try:
            nodes = find_radial_nodes(n, l, Z)
            if len(nodes) > 0:
                print(f"    Radial node positions (a₀): {', '.join([f'{r:.4f}' for r in nodes])}")
        except Exception as e:
            print(f"    (Could not calculate node positions: {e})")
    
    exp_vals = calculate_expectation_values(n, l, Z)
    print(f"{'-'*70}")
    print(f"  Expectation Values (analytical):")
    print(f"    <r> = {exp_vals['<r>']:.6f} a₀")
    print(f"    <r²> = {exp_vals['<r²>']:.6f} a₀²")
    print(f"    <1/r> = {exp_vals['<1/r>']:.6f} a₀⁻¹")
    print(f"{'='*70}\n")

def plot_radial_distribution(n, l, Z, r_max, quality_settings):
    """2D plot of radial probability distribution P(r) = r²|R(r)|²"""
    print(f"\nComputing 2D Radial Distribution for Z={Z}, n={n}, l={l}...")
    num_points = quality_settings['1d_points']
    
    r = xp.linspace(0.001, r_max * a0, num_points, dtype=xp.float64)
    R_vals = R_nl(r, n, l, Z)
    P_vals = r**2 * xp.abs(R_vals)**2
    
    nodes = find_radial_nodes(n, l, Z, r_max)
    
    if hasattr(r, 'get'):
        r_cpu, P_vals_cpu, R_vals_cpu = r.get(), P_vals.get(), R_vals.get()
    else:
        r_cpu, P_vals_cpu, R_vals_cpu = r, P_vals, R_vals
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=r_cpu, y=P_vals_cpu, mode='lines', 
                   name=f'P(r) = r²|R(r)|²', 
                   line=dict(width=3, color='blue')),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=r_cpu, y=R_vals_cpu, mode='lines', 
                   name=f'R(r)', 
                   line=dict(width=2, color='red', dash='dash'),
                   opacity=0.6),
        secondary_y=True
    )
    
    if len(nodes) > 0:
        fig.add_trace(
            go.Scatter(x=nodes, y=np.zeros_like(nodes), 
                       mode='markers', 
                       name='Radial Nodes', 
                       marker=dict(color='red', size=12, symbol='x', line=dict(width=2))),
            secondary_y=False
        )
    
    max_idx = np.argmax(P_vals_cpu)
    r_most_probable = r_cpu[max_idx]
    
    fig.add_vline(x=r_most_probable, line_dash="dot", line_color="green",
                  annotation_text=f"r_max = {r_most_probable:.3f} a₀")
    
    exp_vals = calculate_expectation_values(n, l, Z)
    fig.add_vline(x=exp_vals['<r>'], line_dash="dot", line_color="orange",
                  annotation_text=f"<r> = {exp_vals['<r>']:.3f} a₀")
    
    orbital_label = f"{n}{ORBITAL_LABELS.get(l, '?')}"
    fig.update_layout(
        title=f"Radial Distribution for {orbital_label} Orbital (Z={Z})",
        xaxis_title="Radius r (a₀)",
        yaxis_title="Radial Probability P(r)",
        yaxis2_title="Radial Wavefunction R(r)",
        hovermode='x unified',
        showlegend=True
    )
    
    fig.show()

def visualize_radial_3d_plotly(n, l, Z, r_max, quality_settings):
    """3D volumetric plot of radial probability |R_nl(r)|²"""
    grid_size = quality_settings['3d_grid']
    vol_params = quality_settings['volume']
    print(f"\nComputing 3D radial probability |R_nl|² for Z={Z}, n={n}, l={l} at {grid_size}³ resolution...")

    power_val = quality_settings.get('power_val', 3.0)
    print(f"Using non-uniform 'power' grid with power={power_val} to enhance resolution near origin...")

    with tqdm(total=5, desc="Generating 3D Radial Plot", disable=not TQDM_AVAILABLE) as pbar:
        u = xp.linspace(-1.0, 1.0, grid_size, dtype=xp.float64)
        g = xp.sign(u) * xp.power(xp.abs(u), power_val) * r_max
        grid_key = (grid_size, r_max, f'power_grid_{power_val}')
        X, Y, Z_grid = xp.meshgrid(g, g, g, indexing="ij")
        pbar.set_postfix_str("Grid Created")
        pbar.update(1)

        R = xp.sqrt(X**2 + Y**2 + Z_grid**2)
        pbar.set_postfix_str("Calculating Radii")
        pbar.update(1)

        R_values = R_nl(R, n, l, Z)
        prob_density = xp.abs(R_values)**2
        pbar.set_postfix_str("Density Calculated")
        pbar.update(1)
        
        if hasattr(X, 'get'):
            X_cpu, Y_cpu, Z_cpu, prob_density_cpu = X.get(), Y.get(), Z_grid.get(), prob_density.get()
        else:
            X_cpu, Y_cpu, Z_cpu, prob_density_cpu = X, Y, Z_grid, prob_density
        pbar.set_postfix_str("Data Transferred to CPU")
        pbar.update(1)
        
        if hasattr(X, 'get'):
            del X, Y, Z_grid, R, R_values, prob_density
            cp.get_default_memory_pool().free_all_blocks()
        pbar.update(1)

    max_prob = prob_density_cpu.max()
    if max_prob == 0:
        print("Warning: Maximum probability is zero. Nothing to plot.")
        return

    orbital_label = f"{n}{ORBITAL_LABELS.get(l, '?')}"
    fig = go.Figure(data=go.Volume(
        x=X_cpu.flatten(), y=Y_cpu.flatten(), z=Z_cpu.flatten(), 
        value=prob_density_cpu.flatten(),
        isomin=vol_params['isomin_frac'] * max_prob, 
        isomax=vol_params['isomax_frac'] * max_prob,
        opacity=vol_params['opacity'], 
        surface_count=vol_params['surface_count'], 
        colorscale="Viridis",
        colorbar=dict(title="|R|²")
    ))
    
    fig.update_layout(
        title=f"3D Radial Probability |R<sub>{n},{l}</sub>(r)|² for {orbital_label} (Z={Z})",
        scene=dict(
            xaxis_title="x (a₀)", 
            yaxis_title="y (a₀)", 
            zaxis_title="z (a₀)",
            aspectmode='cube'
        )
    )
    fig.show()

def visualize_angular_plotly(l, m, quality_settings, real=True):
    """3D surface plot of angular distribution |Y_lm(θ,φ)|²"""
    resolution = quality_settings['2d_grid']
    print(f"\nComputing angular distribution for l={l}, m={m} at {resolution}x{resolution} resolution...")
    
    with tqdm(total=4, desc="Generating 3D Angular Plot", disable=not TQDM_AVAILABLE) as pbar:
        theta = xp.linspace(0, np.pi, resolution, dtype=xp.float64)
        phi = xp.linspace(0, 2 * np.pi, resolution, dtype=xp.float64)
        phi_grid, theta_grid = xp.meshgrid(phi, theta)
        pbar.set_postfix_str("Grid Created")
        pbar.update(1)
        
        Y_shape = Y_lm(l, m, theta_grid, phi_grid, real=real)
        r_surface = xp.abs(Y_shape)**2
        pbar.set_postfix_str("Y_lm Calculated")
        pbar.update(1)
        
        x = r_surface * xp.sin(theta_grid) * xp.cos(phi_grid)
        y = r_surface * xp.sin(theta_grid) * xp.sin(phi_grid)
        z = r_surface * xp.cos(theta_grid)
        
        use_phase_color = not real and m != 0
        if use_phase_color:
            Y_complex = Y_lm(l, m, theta_grid, phi_grid, real=False)
            colors = xp.angle(Y_complex)
            colorscale, colorbar_title = 'hsv', "Phase (radians)"
        else:
            colors = r_surface
            colorscale, colorbar_title = 'Viridis', "|Y|²"
        pbar.set_postfix_str("Surface Coordinates Calculated")
        pbar.update(1)

        if hasattr(x, 'get'):
            x_cpu, y_cpu, z_cpu, colors_cpu = x.get(), y.get(), z.get(), colors.get()
        else:
            x_cpu, y_cpu, z_cpu, colors_cpu = x, y, z, colors
        pbar.set_postfix_str("Data Transferred to CPU")
        pbar.update(1)

    surface = go.Surface(
        x=x_cpu, y=y_cpu, z=z_cpu, 
        surfacecolor=colors_cpu, 
        colorscale=colorscale,
        cmin=colors_cpu.min(), cmax=colors_cpu.max(),
        colorbar=dict(title=colorbar_title),
        showscale=True,
        lighting=dict(ambient=0.4, diffuse=0.8, specular=0.5, roughness=0.5),
        lightposition=dict(x=100, y=200, z=50)
    )
    
    title_suffix = "(Real)" if real else "(Complex)"
    if use_phase_color:
        title_suffix += " - Phase Colored"
    
    orbital_type = ORBITAL_LABELS.get(l, '?')
    title = f"Angular Distribution |Y<sub>{l}</sub><sup>{m}</sup>|² ({orbital_type}-orbital) {title_suffix}"
    
    fig = go.Figure(data=[surface])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='x', 
            yaxis_title='y', 
            zaxis_title='z',
            aspectratio=dict(x=1, y=1, z=1),
            camera_eye=dict(x=1.5, y=1.5, z=1.5)
        )
    )
    fig.show()

def plot_2d_slice(n, l, m, Z, r_max, quality_settings, real=True, plane='xy'):
    """2D contour plot of wavefunction probability density"""
    grid_size = quality_settings['2d_grid']
    print(f"\nComputing 2D slice in the {plane}-plane for Z={Z}, n={n}, l={l}, m={m} at {grid_size}x{grid_size} resolution...")
    
    with tqdm(total=5, desc=f"Generating {plane}-plane Slice", disable=not TQDM_AVAILABLE) as pbar:
        g = xp.linspace(-r_max, r_max, grid_size, dtype=xp.float64)
        
        if plane == 'xy':
            X, Y = xp.meshgrid(g, g, indexing='ij')
            Z_grid = xp.zeros_like(X)
            x_label, y_label = "x (a₀)", "y (a₀)"
        elif plane == 'yz':
            Y, Z_grid = xp.meshgrid(g, g, indexing='ij')
            X = xp.zeros_like(Y)
            x_label, y_label = "y (a₀)", "z (a₀)"
        else:      
            X, Z_grid = xp.meshgrid(g, g, indexing='ij')
            Y = xp.zeros_like(X)
            x_label, y_label = "x (a₀)", "z (a₀)"
        
        pbar.set_postfix_str("Grid Created")
        pbar.update(1)

        R = xp.sqrt(X**2 + Y**2 + Z_grid**2)
        R_safe = xp.where(R == 0, 1e-12, R)
        Theta = xp.arccos(xp.clip(Z_grid / R_safe, -1.0, 1.0))
        Phi = xp.arctan2(Y, X)
        pbar.set_postfix_str("Coordinates Transformed")
        pbar.update(1)
        
        psi = psi_nlm(R, Theta, Phi, n, l, m, Z, real=real)
        pbar.set_postfix_str("Wavefunction Calculated")
        pbar.update(1)

        prob_density = xp.abs(psi)**2
        pbar.set_postfix_str("Density Calculated")
        pbar.update(1)

        if hasattr(prob_density, 'get'):
            prob_density_cpu, g_cpu = prob_density.get(), g.get()
        else:
            prob_density_cpu, g_cpu = prob_density, g
        pbar.set_postfix_str("Data Transferred to CPU")
        pbar.update(1)

    orbital_label = f"{n}{ORBITAL_LABELS.get(l, '?')}"
    fig = go.Figure(data=go.Contour(
        z=prob_density_cpu.T,
        x=g_cpu, 
        y=g_cpu,
        colorscale='Viridis',
        contours=dict(coloring='heatmap'),
        colorbar=dict(title='|Ψ|²')
    ))
    
    fig.update_layout(
        title=f"2D Slice |Ψ<sub>{n},{l},{m}</sub>|² on {plane}-plane ({orbital_label}, Z={Z})",
        xaxis_title=x_label,
        yaxis_title=y_label,
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )
    fig.show()

def plot_angular_nodes(l, m, quality_settings):
    """2D map showing angular nodes in (θ, φ) space"""
    resolution = quality_settings['2d_grid']
    print(f"\nComputing 2D Angular Node map for l={l}, m={m} at {resolution}x{resolution} resolution...")
    
    with tqdm(total=3, desc="Generating Angular Node Map", disable=not TQDM_AVAILABLE) as pbar:
        phi = xp.linspace(0, 2 * np.pi, resolution, dtype=xp.float64)
        theta = xp.linspace(0, np.pi, resolution, dtype=xp.float64)
        phi_grid, theta_grid = xp.meshgrid(phi, theta)
        pbar.set_postfix_str("Grid Created")
        pbar.update(1)

        Y_vals = Y_lm(l, m, theta_grid, phi_grid, real=True)
        prob_density = xp.abs(Y_vals)**2
        pbar.set_postfix_str("Density Calculated")
        pbar.update(1)
        
        if hasattr(prob_density, 'get'):
            prob_density_cpu, phi_cpu, theta_cpu = prob_density.get(), phi.get(), theta.get()
        else:
            prob_density_cpu, phi_cpu, theta_cpu = prob_density, phi, theta
        pbar.set_postfix_str("Data Transferred to CPU")
        pbar.update(1)

    orbital_type = ORBITAL_LABELS.get(l, '?')
    fig = go.Figure(data=go.Contour(
        z=prob_density_cpu.T,
        x=phi_cpu, 
        y=theta_cpu,
        colorscale='Viridis',
        contours=dict(coloring='heatmap'),
        colorbar=dict(title='|Y|²')
    ))
    
    fig.update_layout(
        title=f"Angular Nodes for l={l}, m={m} ({orbital_type}-orbital)",
        xaxis_title="Azimuthal Angle φ (radians)",
        yaxis_title="Polar Angle θ (radians)",
        xaxis=dict(
            tickmode='array',
            tickvals=np.linspace(0, 2*np.pi, 5),
            ticktext=['0', 'π/2', 'π', '3π/2', '2π']
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=np.linspace(0, np.pi, 3),
            ticktext=['0', 'π/2', 'π']
        )
    )
    fig.show()

def visualize_total_plotly(n, l, m, Z, r_max, quality_settings, real=True):
    """3D volumetric plot of total probability density |Ψ_nlm|²"""
    grid_size = quality_settings['3d_grid']
    vol_params = quality_settings['volume']
    print(f"\nComputing total probability density for Z={Z}, n={n}, l={l}, m={m} at {grid_size}³ resolution...")
    
    print("Using non-uniform 'power' grid to enhance resolution near origin...")
    power_val = 3.0
    u = xp.linspace(-1.0, 1.0, grid_size, dtype=xp.float64)
    g = xp.sign(u) * xp.power(xp.abs(u), power_val) * r_max
    
    grid_key = (grid_size, r_max, f'power_grid_{power_val}')

    with tqdm(total=5, desc="Generating 3D Total Plot", disable=not TQDM_AVAILABLE) as pbar:
        X, Y, Z_grid = xp.meshgrid(g, g, g, indexing="ij")
        pbar.set_postfix_str("Grid Created")
        pbar.update(1)
        
        R, Theta, Phi = coord_cache.get_spherical(X, Y, Z_grid, grid_key)
        pbar.set_postfix_str("Coordinates Transformed")
        pbar.update(1)

        psi = psi_nlm(R, Theta, Phi, n, l, m, Z, real=real)
        pbar.set_postfix_str("Wavefunction Calculated")
        pbar.update(1)
        
        prob_density = xp.abs(psi)**2

        max_prob = xp.max(prob_density)
        if max_prob == 0:
            print("Warning: Maximum probability is zero. Nothing to plot.")
            return

        floor_prob = max_prob * 1e-7
        prob_density_clipped = xp.maximum(prob_density, floor_prob)

        log_prob_density = xp.log10(prob_density_clipped)
        
        pbar.set_postfix_str("Density Calculated (log-scaled)")
        pbar.update(1)
        
        if hasattr(X, 'get'):
            X_cpu, Y_cpu, Z_cpu, log_prob_density_cpu = X.get(), Y.get(), Z_grid.get(), log_prob_density.get()
            del X, Y, Z_grid, R, Theta, Phi, psi, prob_density, prob_density_clipped, log_prob_density
            cp.get_default_memory_pool().free_all_blocks()
        else:
            X_cpu, Y_cpu, Z_cpu, log_prob_density_cpu = X, Y, Z_grid, log_prob_density
        
        pbar.set_postfix_str("Data Transferred to CPU")
        pbar.update(1)

    log_max = log_prob_density_cpu.max()
    log_min = log_prob_density_cpu.min()                                
    log_range = log_max - log_min

    new_isomin = log_min + vol_params['isomin_frac'] * log_range
    new_isomax = log_min + vol_params['isomax_frac'] * log_range

    orbital_label = f"{n}{ORBITAL_LABELS.get(l, '?')}"
    title_suffix = "(Real)" if real else "(Complex)"
    
    fig = go.Figure(data=go.Volume(
        x=X_cpu.flatten(), y=Y_cpu.flatten(), z=Z_cpu.flatten(),

        value=log_prob_density_cpu.flatten(),                          
        isomin=new_isomin,                                                
        isomax=new_isomax,                                                

        opacity=vol_params['opacity'],
        surface_count=vol_params['surface_count'],
        colorscale="Viridis",

        colorbar=dict(title="log₁₀(|Ψ|²)")                                

    ))
    
    fig.update_layout(
        
        title=f"Total Probability Density log₁₀(|Ψ<sub>{n},{l},{m}</sub>|²) {title_suffix} ({orbital_label}, Z={Z})",                   
        
        scene=dict(
            xaxis_title="x (a₀)",
            yaxis_title="y (a₀)",
            zaxis_title="z (a₀)",
            aspectmode='cube'
        )
    )
    fig.show()

def create_superposition_animation(n_list, l_list, m_list, coeffs, Z, r_max, quality_settings, num_frames=50):
    """
    Create animation of time-dependent superposition state
    
    Ψ(t) = Σ c_i * ψ_i * exp(-i E_i t / ℏ)
    """
    print("\n" + "="*70)
    print("Creating Time-Dependent Superposition Animation")
    print("="*70)
    
    grid_size = quality_settings['2d_grid']
    
    g = xp.linspace(-r_max, r_max, grid_size, dtype=xp.float64)
    X, Y = xp.meshgrid(g, g, indexing='ij')
    Z_grid = xp.zeros_like(X)
    
    R = xp.sqrt(X**2 + Y**2 + Z_grid**2)
    R_safe = xp.where(R == 0, 1e-12, R)
    Theta = xp.arccos(xp.clip(Z_grid / R_safe, -1.0, 1.0))
    Phi = xp.arctan2(Y, X)
    
    psi_states = []
    energies = []
    
    print("Computing individual states...")
    for i, (n, l, m) in enumerate(zip(n_list, l_list, m_list)):
        psi = psi_nlm(R, Theta, Phi, n, l, m, Z, real=False)
        psi_states.append(psi)
        E_hartree, _ = calculate_energy(n, Z)
        energies.append(E_hartree)
        print(f"  State {i+1}: n={n}, l={l}, m={m}, E={E_hartree:.6f} Ha")
    
    coeffs = np.array(coeffs, dtype=np.complex128)
    coeffs = coeffs / np.sqrt(np.sum(np.abs(coeffs)**2))
    
    energies_np = np.array(energies)
    E_diffs = np.abs(energies_np - energies_np[:, None])
    min_E_diff = np.min(E_diffs[np.nonzero(E_diffs)])
    T_period = 2 * np.pi / min_E_diff
    
    times = xp.linspace(0, T_period, num_frames)
    
    print(f"Generating {num_frames} animation frames (Period T = {T_period:.2f} a.u.)...")
    
    z_max = 0
    frame_data = []

    for t in tqdm(times, desc="Animation Progress", disable=not TQDM_AVAILABLE):
        psi_t = xp.zeros_like(psi_states[0], dtype=xp.complex128)
        for i, (psi_i, E_i, c_i) in enumerate(zip(psi_states, energies, coeffs)):
            phase = xp.exp(-1j * E_i * t)
            psi_t += c_i * psi_i * phase
        
        prob_density = xp.abs(psi_t)**2
        
        if hasattr(prob_density, 'get'):
            prob_cpu = prob_density.get()
        else:
            prob_cpu = prob_density
        
        frame_data.append(prob_cpu.T)
        z_max = max(z_max, prob_cpu.max())
    
    if hasattr(g, 'get'):
        g_cpu = g.get()
        times_cpu = times.get()
    else:
        g_cpu = g
        times_cpu = times
    
    frames = []
    for k, data in enumerate(frame_data):
        frames.append(go.Frame(
            data=[go.Contour(z=data, x=g_cpu, y=g_cpu,
                             colorscale='Viridis', zmin=0, zmax=z_max,
                             contours=dict(coloring='heatmap'))],
            name=f"t={times_cpu[k]:.3f}"
        ))
    
    fig = go.Figure(
        data=[go.Contour(z=frame_data[0], x=g_cpu, y=g_cpu, 
                         colorscale='Viridis', zmin=0, zmax=z_max,
                         contours=dict(coloring='heatmap'),
                         colorbar=dict(title='|Ψ(t)|²'))],
        layout=go.Layout(
            title="Time Evolution of Superposition State (xy-plane)",
            xaxis=dict(title="x (a₀)"),
            yaxis=dict(title="y (a₀)", scaleanchor="x", scaleratio=1),
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(label="Play", method="animate", 
                         args=[None, {"frame": {"duration": 100, "redraw": True}, 
                                       "fromcurrent": True}]),
                    dict(label="Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False}, 
                                        "mode": "immediate"}])
                ]
            )]
        ),
        frames=frames
    )
    
    fig.show()
    print("Animation complete!")

def compare_orbitals(orbital_list, Z, r_max, quality_settings):
    """Compare multiple orbitals side-by-side"""
    print("\n" + "="*70)
    print("Comparing Multiple Orbitals")
    print("="*70)
    
    num_orbitals = len(orbital_list)
    grid_size = quality_settings['2d_grid']
    
    cols = min(num_orbitals, 3)
    rows = int(np.ceil(num_orbitals / cols))
    
    subplot_titles = [f"n={n}, l={l}, m={m} ({n}{ORBITAL_LABELS.get(l, '?')})" for n, l, m in orbital_list]
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles,
        specs=[[{'type': 'contour'} for _ in range(cols)] for _ in range(rows)]
    )
    
    g = xp.linspace(-r_max, r_max, grid_size, dtype=xp.float64)
    X, Z_grid = xp.meshgrid(g, g, indexing='ij')
    Y = xp.zeros_like(X)
    
    R = xp.sqrt(X**2 + Y**2 + Z_grid**2)
    R_safe = xp.where(R == 0, 1e-12, R)
    Theta = xp.arccos(xp.clip(Z_grid / R_safe, -1.0, 1.0))
    Phi = xp.arctan2(Y, X)
    
    if hasattr(g, 'get'):
        g_cpu = g.get()
    else:
        g_cpu = g
    
    max_z = 0
    prob_data = []

    print("Computing orbitals for comparison...")
    for idx, (n, l, m) in enumerate(tqdm(orbital_list, disable=not TQDM_AVAILABLE)):
        psi = psi_nlm(R, Theta, Phi, n, l, m, Z, real=True)
        prob_density = xp.abs(psi)**2
        
        if hasattr(prob_density, 'get'):
            prob_cpu = prob_density.get()
        else:
            prob_cpu = prob_density
        
        prob_data.append(prob_cpu.T)
        max_z = max(max_z, prob_cpu.max())

    print("Generating plots...")
    for idx, (n, l, m) in enumerate(orbital_list):
        row = idx // cols + 1
        col = idx % cols + 1
        
        fig.add_trace(
            go.Contour(z=prob_data[idx], x=g_cpu, y=g_cpu, colorscale='Viridis',
                       contours=dict(coloring='heatmap'),
                       zmin=0, zmax=max_z,
                       showscale=(idx == 0),
                       colorbar=dict(title='|Ψ|²') if idx == 0 else None),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="x (a₀)", row=row, col=col)
        fig.update_yaxes(title_text="z (a₀)", row=row, col=col, scaleanchor=f"x{idx+1}", scaleratio=1)
    
    fig.update_layout(
        title_text=f"Orbital Comparison (xz-plane slices, Z={Z})",
        height=400 * rows,
        showlegend=False
    )
    
    fig.show()

def plot_energy_levels(Z, n_max=5):
    """Plot energy level diagram with transitions"""
    print(f"\nPlotting energy levels for Z={Z}...")
    
    fig = go.Figure()
    
    colors = ['#EF5350', '#42A5F5', '#66BB6A', '#FFA726', '#AB47BC', '#8D6E63', '#26A69A']
    
    for n in range(1, n_max + 1):
        E_hartree, E_eV = calculate_energy(n, Z)
        
        for l in range(n):
            orbital_label = f"{n}{ORBITAL_LABELS.get(l, '?')}"
            
            x_pos = l
            x_start = x_pos - 0.4
            x_end = x_pos + 0.4
            
            fig.add_trace(go.Scatter(
                x=[x_start, x_end],
                y=[E_eV, E_eV],
                mode='lines',
                line=dict(color=colors[(n-1) % len(colors)], width=4),
                name=f"n={n}",
                legendgroup=f"n={n}",
                showlegend=(l == 0)
            ))
            
            fig.add_annotation(
                x=x_pos,
                y=E_eV,
                text=orbital_label,
                showarrow=False,
                yshift=10,
                font=dict(size=10)
            )

    if n_max >= 2:
        E1, E2 = calculate_energy(1, Z)[1], calculate_energy(2, Z)[1]
        fig.add_trace(go.Scatter(
            x=[1, 0], y=[E2, E1], 
            mode='lines+markers', 
            line=dict(color='purple', width=1, dash='dash'), 
            name='Lyman α', 
            marker=dict(symbol='arrow', size=8, angleref='previous')
        ))
    
    if n_max >= 3:
        E2, E3 = calculate_energy(2, Z)[1], calculate_energy(3, Z)[1]
        fig.add_trace(go.Scatter(
            x=[2, 1], y=[E3, E2], 
            mode='lines+markers', 
            line=dict(color='red', width=1, dash='dash'), 
            name='Balmer α', 
            marker=dict(symbol='arrow', size=8, angleref='previous')
        ))

    fig.update_layout(
        title=f"Energy Level Diagram for Z={Z}",
        xaxis_title="Angular Momentum l",
        yaxis_title="Energy (eV)",
        xaxis=dict(tickmode='array', tickvals=list(range(n_max)), 
                   ticktext=[f"l={l} ({ORBITAL_LABELS.get(l, str(l))})" for l in range(n_max)]),
        hovermode='closest',
        height=600
    )
    
    fig.show()

def plot_correction_comparison(n, Z):
    """
    Generate an interactive, hierarchical splitting diagram for a 
    single principal quantum number 'n', similar to textbook examples.
    """
    print(f"\nGenerating hierarchical splitting diagram for n={n}, Z={Z}...")

    fig = go.Figure()

    categories = [
        'Bohr (n)', 
        'Bohr + Relativistic (n,l)', 
        'Bohr + Rel + Darwin (n,l)', 
        'Full Fine Structure (n,l,j)', 
        'Full FS + Hyperfine (n,l,j,F)'
    ]
    
    colors = ['#EF5350', '#42A5F5', '#66BB6A', '#FFA726', '#AB47BC', '#8D6E63', '#26A69A']
    color_index = 0

    for l in range(n):
                               
        _, E_Bohr_val = calculate_energy(n, Z)

        dE_Rel = relativistic_correction(n, l, Z)
        E_Rel_val = E_Bohr_val + dE_Rel

        dE_Dar = darwin_correction(n, l, Z)
        E_Dar_val = E_Rel_val + dE_Dar
        
        if l == 0: 
            j_states = [0.5]
        else: 
            j_states = [l - 0.5, l + 0.5]
            j_states.sort()
        
        for j in j_states:

            dE_FS = calculate_fine_structure(n, l, j, Z)
            E_FS_val = E_Bohr_val + dE_FS

            F_states = [j + I_proton]
            if (j - I_proton >= 0) and not np.isclose(j - I_proton, j + I_proton):
                F_states.append(j - I_proton)
            F_states = sorted(list(set(F_states)))
            
            for F in F_states:
                                            
                dE_HFS = hyperfine_splitting(n, l, j, F, Z)
                E_HFS_val = E_FS_val + dE_HFS
                
                state_label = f"{n}{ORBITAL_LABELS.get(l, '?')}<sub>{j:.1f}</sub> (F={F:.1f})"
                
                x_data = [categories[0], categories[1], categories[2], categories[3]]
                y_data = [E_Bohr_val, E_Rel_val, E_Dar_val, E_FS_val]

                if abs(dE_HFS) > 1e-15:
                    x_data.append(categories[4])
                    y_data.append(E_HFS_val)
                else:
                                                                                
                    x_data.append(categories[4])
                    y_data.append(E_FS_val)                      

                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='lines+markers',
                    name=state_label,
                    legendgroup=f"l={l}",
                    line=dict(color=colors[color_index % len(colors)]),
                    hovertemplate=f"<b>{state_label}</b><br>Level: %{ x} <br>Energy: %{ y:.12f}  eV<extra></extra>"
                ))
            
            color_index += 1                                

    fig.update_layout(
        title=f"Hierarchical Energy Level Splitting for n={n} (Z={Z})",
        xaxis_title="Correction Level",
        yaxis_title="Energy (eV)",
        xaxis=dict(type='category', tickangle=-15),
        yaxis=dict(
                                                                            
            tickformat=".6f",
            hoverformat=".12f"
        ),
        hovermode='closest',
        height=700,
        legend_title_text="Quantum State (n l<sub>j</sub> F)"
    )
    
    print("Displaying interactive splitting diagram...")
    fig.show()

def bohr_energy(n, Z):
    """Returns the Bohr energy in eV"""
    _, E_eV = calculate_energy(n, Z)
    return E_eV

def main_menu():
    """Main interactive menu"""
    print("\n" + "="*70)
    print(" HYDROGEN-LIKE ATOM ORBITAL VISUALIZATION TOOLKIT")
    print(" (Enhanced with Physics Validation & Advanced Features)")
    print("="*70)

    if cp:
        use_gpu_val = 'y' in input("Use GPU (CuPy) if available? (y/n): ").lower()
        set_backend(use_gpu_val)
    else:
        set_backend(False)

    quality_presets = {
        '1': {'name': 'Low',    '1d_points': 1000, '2d_grid': 120, '3d_grid': 40,
              'volume': {'isomin_frac': 0.1, 'isomax_frac': 0.8, 'opacity': 0.1, 'surface_count': 10}},
        '2': {'name': 'Medium', '1d_points': 2000, '2d_grid': 200, '3d_grid': 60,
              'volume': {'isomin_frac': 0.05, 'isomax_frac': 1.0, 'opacity': 0.08, 'surface_count': 35}},
        '3': {'name': 'High',   '1d_points': 4000, '2d_grid': 300, '3d_grid': 80,
              'volume': {'isomin_frac': 0.05, 'isomax_frac': 0.9, 'opacity': 0.05, 'surface_count': 50}},
        '4': {'name': 'Ultra',  '1d_points': 8000, '2d_grid': 400, '3d_grid': 128,
              'volume': {'isomin_frac': 0.05, 'isomax_frac': 0.9, 'opacity': 0.05, 'surface_count': 50}},
    }
    
    while True:
        try:
            print("\n" + "="*70)
            Z_val = int(input("Enter atomic number Z (e.g., 1 for H, 2 for He+): ") or 1)
            if Z_val < 1:
                print("Atomic number Z must be >= 1. Defaulting to 1.")
                Z_val = 1
            
            n_val = int(input("Enter principal quantum number n (e.g., 3): ") or 3)
            if n_val < 1:
                raise ValueError("n must be >= 1")
            
            l_val = int(input(f"Enter angular quantum number l (0 to {n_val-1}): ") or 0)
            if not (0 <= l_val < n_val):
                print(f"l must be in [0, {n_val-1}]. Defaulting to 0.")
                l_val = 0
            
            m_val = int(input(f"Enter magnetic quantum number m (-{l_val} to {l_val}): ") or 0)
            if not (-l_val <= m_val <= l_val):
                print(f"m must be in [{-l_val}, {l_val}]. Defaulting to 0.")
                m_val = 0
            
            use_real_val = 'y' in input("Use real spherical harmonics for plots? (y/n): ").lower()
            
            print("\nSelect a plot quality/resolution level:")
            quality_choice = input("[1] Low\n[2] Medium\n[3] High\n[4] Ultra (GPU Recommended)\nEnter choice [2]: ") or '2'
            settings = quality_presets.get(quality_choice, quality_presets['2'])
            print(f"Using '{settings['name']}' quality settings.")
            
        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")
            continue
        except KeyboardInterrupt:
            print("\n\nExiting visualization toolkit. Goodbye!")
            return
        
        r_max_val = n_val * (n_val + 15) / Z_val
        
        show_fs = 'y' in input("\nShow fine structure corrections? (y/n): ").lower()
        print_orbital_info(n_val, l_val, m_val, Z_val, show_fine_structure=show_fs)
        
        if 'y' in input("Verify wavefunction normalization? (y/n): ").lower():
            try:
                norm_integral = check_normalization(n_val, l_val, m_val, Z_val)
                print(f"Normalization integral: {norm_integral:.6f} (should be ≈ 1.0)")
                if abs(norm_integral - 1.0) > 0.1:
                    print("WARNING: Large deviation from unity. Check implementation!")
                else:
                    print("✓ Normalization check passed!")
            except Exception as e:
                print(f"Error checking normalization: {e}")

        while True:
            print("\n" + "="*70)
            print(f"Current: Z={Z_val} | n={n_val}, l={l_val}, m={m_val} | Quality: {settings['name']}")
            print("="*70)
            print("BASIC VISUALIZATIONS:")
            print("  [1] 2D Radial Distribution P(r)")
            print("  [2] 3D Radial Probability |R_nl|²")
            print("  [3] 3D Angular Distribution |Y_lm|²")
            print("  [4] 2D Planar Slices of |Ψ|²")
            print("  [5] 2D Angular Node Map")
            print("  [6] 3D Total Probability |Ψ|²")
            print("\nADVANCED FEATURES:")
            print("  [7] Energy Level Diagram")
            print("  [8] Compare Multiple Orbitals")
            print("  [9] Time-Dependent Superposition Animation")
            print("  [10] Plot Hierarchical Energy Splitting Diagram (for current n)")
            print("\n  [q] Change quantum numbers or exit")
            
            choice = input("\nEnter your choice: ").lower()

            try:
                if choice == '1':
                    plot_radial_distribution(n_val, l_val, Z_val, r_max=r_max_val, quality_settings=settings)
                
                elif choice == '2':
                    visualize_radial_3d_plotly(n_val, l_val, Z_val, r_max=r_max_val, quality_settings=settings)
                
                elif choice == '3':
                    visualize_angular_plotly(l_val, m_val, quality_settings=settings, real=use_real_val)
                
                elif choice == '4':
                    plane = input("Enter plane for slice ('xy', 'yz', or 'xz') [xz]: ").lower() or 'xz'
                    if plane not in ['xy', 'yz', 'xz']:
                        plane = 'xz'
                    plot_2d_slice(n_val, l_val, m_val, Z_val, r_max=r_max_val, 
                                 quality_settings=settings, real=use_real_val, plane=plane)
                
                elif choice == '5':
                    plot_angular_nodes(l_val, m_val, quality_settings=settings)
                
                elif choice == '6':
                    visualize_total_plotly(n_val, l_val, m_val, Z_val, r_max=r_max_val,
                                          quality_settings=settings, real=use_real_val)
                
                elif choice == '7':
                    n_max = int(input("Enter maximum n to display (e.g., 5): ") or 5)
                    plot_energy_levels(Z_val, n_max=n_max)
                
                elif choice == '8':
                    print("Enter orbitals to compare (n l m), one per line. Empty line to finish:")
                    orbitals = []
                    while True:
                        line = input(f"Orbital {len(orbitals)+1} (or Enter to finish): ").strip()
                        if not line:
                            break
                        try:
                            n, l, m = map(int, line.split())
                            if 0 <= l < n and -l <= m <= l:
                                orbitals.append((n, l, m))
                            else:
                                print("Invalid quantum numbers! (n>l, |m|<=l)")
                        except ValueError:
                            print("Invalid format! Use: n l m")
                    
                    if len(orbitals) > 0:
                        compare_orbitals(orbitals, Z_val, r_max_val, settings)
                    else:
                        print("No orbitals to compare.")
                
                elif choice == '9':
                    print("Create superposition of states:")
                    print("Enter quantum numbers and coefficients (n l m real_coeff imag_coeff)")
                    print("Empty line to finish:")
                    n_list, l_list, m_list, coeffs = [], [], [], []
                    
                    while True:
                        line = input(f"State {len(n_list)+1} (or Enter to finish): ").strip()
                        if not line:
                            break
                        try:
                            parts = line.split()
                            n, l, m = map(int, parts[:3])
                            real_c = float(parts[3]) if len(parts) > 3 else 1.0
                            imag_c = float(parts[4]) if len(parts) > 4 else 0.0
                            
                            if 0 <= l < n and -l <= m <= l:
                                n_list.append(n)
                                l_list.append(l)
                                m_list.append(m)
                                coeffs.append(complex(real_c, imag_c))
                            else:
                                print("Invalid quantum numbers! (n>l, |m|<=l)")
                        except (ValueError, IndexError):
                            print("Invalid format! Use: n l m [real_coeff] [imag_coeff]")
                    
                    if len(n_list) >= 2:
                        num_frames = int(input("Enter number of animation frames (e.g., 50): ") or 50)
                        create_superposition_animation(n_list, l_list, m_list, coeffs, Z_val, r_max_val, settings, num_frames)
                    elif len(n_list) > 0:
                        print("Superposition requires at least two states. Aborting.")
                    else:
                        print("No states entered for superposition.")
                
                elif choice == '10':
                    print(f"Generating splitting diagram for current n={n_val}...")
                    plot_correction_comparison(n_val, Z_val)

                elif choice == 'q':
                    print("\nReturning to main selection...")
                    coord_cache.clear()
                    break
                
                else:
                    print(f"Invalid choice '{choice}'. Please try again.")
                
            except KeyboardInterrupt:
                print("\n\nReturning to main menu...")
                break
            except Exception as e:
                print(f"\n" + "!"*70)
                print(f"  An error occurred: {e}")
                print(f"  Please check your inputs or quality settings.")
                print("!"*70 + "\n")
                traceback.print_exc()
        
        if choice == 'q':
            if 'n' in input("Do you want to select new quantum numbers? (y/n): ").lower():
                print("\nExiting visualization toolkit. Goodbye!")
                break

if __name__ == "__main__":
    coord_cache.clear()
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Goodbye!")
    except Exception as e:
        print(f"\nFatal error: {e}")
        traceback.print_exc()