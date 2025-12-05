           
import numpy as np
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS

try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    print("Warning: 'scikit-image' not found. 3D Total Plot will not work.")
    print("Please install it: pip install scikit-image")
    SKIMAGE_AVAILABLE = False

try:
    from scipy.ndimage import map_coordinates
    SCIPY_NDIMAGE_AVAILABLE = True
except ImportError:
    print("Warning: 'scipy' not found. 3D phase coloring will not work.")
    print("Please install it: pip install scipy")
    SCIPY_NDIMAGE_AVAILABLE = False

try:
    from Hydrogen_like_atom import (
        set_backend, cp, xp,
        psi_nlm, R_nl, Y_lm,
        calculate_energy, calculate_fine_structure,
        find_radial_nodes, calculate_expectation_values,
        print_orbital_info, ORBITAL_LABELS,
        relativistic_correction, darwin_correction, spin_orbit_correction,
        hyperfine_splitting, m_e_c2_eV, I_proton, s_electron,
        check_normalization, HARTREE_TO_EV                  
    )
    print("Successfully imported Hydrogen_like_atom.py library.")
except ImportError as e:
    print(f"FATAL ERROR: Could not import 'Hydrogen_like_atom.py'. {e}")
    print("Please make sure your original script is saved in the same directory.")
    exit()

if cp:
    set_backend(True)
else:
    set_backend(False)

app = Flask(__name__)
CORS(app)                                     

def get_3d_grid_coords(grid_size, r_max):
    """Helper to generate a 3D coordinate grid on the GPU/CPU"""
    dtype = xp.float32
    power_val = 3.0
    u = xp.linspace(-1.0, 1.0, grid_size, dtype=dtype)
    g = xp.sign(u) * xp.power(xp.abs(u), power_val) * r_max

    X, Y, Z_grid = xp.meshgrid(g, g, g, indexing="ij")

    R = xp.sqrt(X**2 + Y**2 + Z_grid**2)
    R_safe = xp.where(R == 0, 1e-9, R)
    Theta = xp.arccos(xp.clip(Z_grid / R_safe, -1.0, 1.0))
    Phi = xp.arctan2(Y, X)

    return X, Y, Z_grid, R, Theta, Phi, g

def compute_3d_isosurfaces(surface_grid, color_grid, min_val, max_val, grid_coords, num_surfaces=60):                           
    """
    Core function to compute multiple isosurfaces (marching cubes).
    - surface_grid: The grid to calculate the *shape* of the surfaces (e.g., log-prob)
    - color_grid: The grid to sample *colors* from (e.g., phase/sign)
    """
    if not SCIPY_NDIMAGE_AVAILABLE:
        raise ImportError("Scipy.ndimage is required for 3D interpolation.")

    print(f"Generating {num_surfaces} isosurfaces...")
    meshes = []

    levels = np.linspace(min_val + (max_val - min_val) * 0.1, min_val + (max_val - min_val) * 0.9, num_surfaces)

    base_opacities_raw = np.linspace(0.0, 1.0, num_surfaces)         
    opacities = 0.01 + 0.24 * np.power(base_opacities_raw, 1.5)                                        

    if hasattr(surface_grid, 'get'):
        surface_grid_cpu = surface_grid.get()
        del surface_grid
    else:
        surface_grid_cpu = surface_grid

    if hasattr(color_grid, 'get'):
        color_grid_cpu = color_grid.get()
        del color_grid
    else:
        color_grid_cpu = color_grid

    if hasattr(xp, 'get_default_memory_pool'):
         cp.get_default_memory_pool().free_all_blocks()

    g_cpu = grid_coords
    if hasattr(g_cpu, 'get'):
        g_cpu = g_cpu.get()

    res = len(g_cpu)
    r_max_abs = np.max(np.abs(g_cpu))                          

    for level, base_opacity in zip(levels, opacities):
        try:
            verts, faces, _, _ = measure.marching_cubes(
                surface_grid_cpu,
                level=level
            )

            vert_colors = map_coordinates(color_grid_cpu, verts.T, order=1, mode='nearest')

            scaled_verts = np.interp(verts.flatten(), np.arange(res), g_cpu).reshape(verts.shape)

            x, y, z = scaled_verts[:, 0], scaled_verts[:, 1], scaled_verts[:, 2]

            r_coords = np.sqrt(x**2 + y**2 + z**2)
            r_avg = np.mean(r_coords) if len(r_coords) > 0 else 0

            if r_max_abs > 1e-9:
                                                         
                 radial_scale = np.clip(1.0 - (r_avg / r_max_abs), 0.05, 1.0)**3                      
            else:
                 radial_scale = 1.0
            final_opacity = base_opacity * radial_scale

            i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

            meshes.append({
                "x": x.tolist(), "y": y.tolist(), "z": z.tolist(),
                "i": i.tolist(), "j": j.tolist(), "k": k.tolist(),
                "opacity": final_opacity,                             
                "intensity": vert_colors.tolist(),
            })
        except ValueError:                                                     
             print(f"Warning: No surface found for level {level}. Skipping.")
        except Exception as e:
            print(f"Warning: Marching cubes failed for level {level}: {e}")

    print("Isosurface generation complete.")
    return meshes

@app.route('/api/orbital-info', methods=['GET'])
def get_orbital_info():
    """Replaces print_orbital_info()"""
    try:
        n = int(request.args.get('n'))
        l = int(request.args.get('l'))
        m = int(request.args.get('m'))
        Z = int(request.args.get('Z'))
        show_fs = request.args.get('fs', 'false').lower() == 'true'

        info = {}
        info['n'], info['l'], info['m'], info['Z'] = n, l, m, Z
        info['label'] = f"{n}{ORBITAL_LABELS.get(l, '?')}"

        E_hartree, E_eV = calculate_energy(n, Z)
        info['energy_hartree'] = f"{E_hartree:.8f}"
        info['energy_ev'] = f"{E_eV:.6f}"

        info['nodes_radial'] = n - l - 1
        info['nodes_angular'] = l
        info['nodes_total'] = n - 1

        if info['nodes_radial'] > 0:
            nodes = find_radial_nodes(n, l, Z)
            info['node_positions'] = [f"{r:.4f} a₀" for r in nodes]

        exp_vals = calculate_expectation_values(n, l, Z)
        info['exp_r'] = f"{exp_vals['<r>']:.6f} a₀"
        info['exp_r2'] = f"{exp_vals['<r²>']:.6f} a₀²"
        info['exp_r_inv'] = f"{exp_vals['<1/r>']:.6f} a₀⁻¹"

        if show_fs:
            info['fs'] = []
            if l == 0:
                j = 0.5
                E_fs = calculate_fine_structure(n, l, j, Z)
                info['fs'].append(f"j = 0.5: ΔE = {E_fs:.6e} eV")
            else:
                j_minus, j_plus = l - 0.5, l + 0.5
                E_fs_minus = calculate_fine_structure(n, l, j_minus, Z)
                E_fs_plus = calculate_fine_structure(n, l, j_plus, Z)
                info['fs'].append(f"j = {j_minus}: ΔE = {E_fs_minus:.6e} eV")
                info['fs'].append(f"j = {j_plus}: ΔE = {E_fs_plus:.6e} eV")

        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route('/api/check-normalization', methods=['GET'])
def get_normalization():
    """(NEW) Replaces check_normalization()"""
    try:
        n = int(request.args.get('n'))
        l = int(request.args.get('l'))
        m = int(request.args.get('m'))
        Z = int(request.args.get('Z'))

        integral = check_normalization(n, l, m, Z, n_samples=100000)

        return jsonify({
            "integral_value": float(integral),
            "passed": bool(abs(integral - 1.0) < 0.1)
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route('/api/radial-distribution', methods=['GET'])
def get_radial_distribution():
    """Replaces plot_radial_distribution()"""
    try:
        n = int(request.args.get('n'))
        l = int(request.args.get('l'))
        Z = int(request.args.get('Z'))
        r_max = float(request.args.get('r_max'))
        res = int(request.args.get('res'))

        r = xp.linspace(0.001, r_max, res, dtype=xp.float32)
        R_vals = R_nl(r, n, l, Z)
        P_vals = r**2 * xp.abs(R_vals)**2

        nodes = find_radial_nodes(n, l, Z)
        exp_vals = calculate_expectation_values(n, l, Z)

        if hasattr(r, 'get'):
            r_cpu, P_cpu, R_cpu = r.get(), P_vals.get(), R_vals.get()
        else:
            r_cpu, P_cpu, R_cpu = r, P_vals, R_vals

        r_most_probable = r_cpu[np.argmax(P_cpu)] if len(P_cpu) > 0 else 0

        return jsonify({
            "r": r_cpu.tolist(), "P_r": P_cpu.tolist(), "R_r": R_cpu.tolist(),
            "nodes": nodes.tolist(), "r_most_probable": float(r_most_probable),
            "exp_r": float(exp_vals['<r>'])
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route('/api/radial-isosurface', methods=['GET'])
def get_radial_isosurface():
    """(MODIFIED) Uses probability for both shape and color"""
    if not SKIMAGE_AVAILABLE:
        return jsonify({"error": "scikit-image library not found on server."}), 500
    try:
        n = int(request.args.get('n'))
        l = int(request.args.get('l'))
        Z = int(request.args.get('Z'))
        res = int(request.args.get('res'))
        r_max = float(request.args.get('r_max'))

        _, _, _, R, _, _, g = get_3d_grid_coords(res, r_max)

        R_vals = R_nl(R, n, l, Z)
        prob_density = xp.abs(R_vals)**2

        min_val, max_val = float(xp.min(prob_density)), float(xp.max(prob_density))
        if max_val <= 1e-12:                            
            return jsonify({"error": "Max probability is effectively zero."}), 500

        meshes = compute_3d_isosurfaces(prob_density, prob_density, min_val, max_val, g)

        return jsonify({
            "meshes": meshes,
            "cmin": min_val, "cmax": max_val,
            "colorscale": "Viridis", "colorbar_title": "|R|²"
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route('/api/angular-isosurface', methods=['GET'])
def get_angular_isosurface():
    """(REWRITTEN) Calculates shape from |Y| but color from phase(Y)"""
    if not SKIMAGE_AVAILABLE:
        return jsonify({"error": "scikit-image library not found on server."}), 500
    try:
        l = int(request.args.get('l'))
        m = int(request.args.get('m'))
        res = int(request.args.get('res'))
        real = request.args.get('real', 'true').lower() == 'true'

        angular_scale = 1.5
        _, _, _, _, Theta, Phi, g = get_3d_grid_coords(res, angular_scale)

        Y_vals = Y_lm(l, m, Theta, Phi, real=real)

        if real:
            color_grid = xp.sign(Y_vals)
            colorscale = 'RdBu'                     
            cmin, cmax = -1.0, 1.0
            colorbar_title = 'Sign (+/-)'
        else:
                                                             
            color_grid = xp.angle(Y_vals + 1e-15)
            colorscale = 'hsv'           
            cmin, cmax = -np.pi, np.pi
            colorbar_title = 'Phase (rad)'

        surface_grid = xp.abs(Y_vals)
        surf_min, surf_max = float(xp.min(surface_grid)), float(xp.max(surface_grid))
        if surf_max <= 1e-12:
            return jsonify({"error": "Max probability is effectively zero."}), 500

        meshes = compute_3d_isosurfaces(surface_grid, color_grid, surf_min, surf_max, g)

        return jsonify({
            "meshes": meshes,
            "cmin": cmin, "cmax": cmax,
            "colorscale": colorscale, "colorbar_title": colorbar_title,
            "fixed_scale": angular_scale                               
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500
    
@app.route('/api/angular-surface-plot', methods=['GET'])
def get_angular_surface_plot():
    """(NEW) Generates a single 'balloon' surface plot, not an isosurface."""
    try:
        l = int(request.args.get('l'))
        m = int(request.args.get('m'))
        res = int(request.args.get('res'))
        real = request.args.get('real', 'true').lower() == 'true'

        surface_res = max(res * 2, 100)                          
        theta = xp.linspace(0, np.pi, surface_res, dtype=xp.float32)
        phi = xp.linspace(0, 2 * np.pi, surface_res, dtype=xp.float32)
        phi_grid, theta_grid = xp.meshgrid(phi, theta)

        Y_vals = Y_lm(l, m, theta_grid, phi_grid, real=real)

        r_surface = xp.abs(Y_vals)**2

        x_surf = r_surface * xp.sin(theta_grid) * xp.cos(phi_grid)
        y_surf = r_surface * xp.sin(theta_grid) * xp.sin(phi_grid)
        z_surf = r_surface * xp.cos(theta_grid)

        if real:
            colors = xp.sign(Y_vals)
            colorscale = 'RdBu'                     
            cmin, cmax = -1.0, 1.0
            colorbar_title = 'Sign (+/-)'
        else:
            colors = xp.angle(Y_vals + 1e-15)              
            colorscale = 'hsv'           
            cmin, cmax = -np.pi, np.pi
            colorbar_title = 'Phase (rad)'

        if hasattr(x_surf, 'get'):
            x_cpu = x_surf.get().tolist()
            y_cpu = y_surf.get().tolist()
            z_cpu = z_surf.get().tolist()
            colors_cpu = colors.get().tolist()
        else:
            x_cpu = x_surf.tolist()
            y_cpu = y_surf.tolist()
            z_cpu = z_surf.tolist()
            colors_cpu = colors.tolist()

        return jsonify({
            "x": x_cpu, "y": y_cpu, "z": z_cpu,
            "surfacecolor": colors_cpu,                                          
            "cmin": cmin, "cmax": cmax,
            "colorscale": colorscale, "colorbar_title": colorbar_title
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route('/api/2d-slice', methods=['GET'])
def get_2d_slice():
    """Replaces plot_2d_slice()"""
    try:
        n = int(request.args.get('n'))
        l = int(request.args.get('l'))
        m = int(request.args.get('m'))
        Z = int(request.args.get('Z'))
        real = request.args.get('real', 'true').lower() == 'true'
        plane = request.args.get('plane', 'xz')
        position = float(request.args.get('pos', 0.0))
        res = int(request.args.get('res'))
        r_max = float(request.args.get('r_max'))

        g = xp.linspace(-r_max, r_max, res, dtype=xp.float32)

        if plane == 'xy':
            X, Y = xp.meshgrid(g, g, indexing='ij')
            Z_grid = xp.full_like(X, fill_value=position)
            x_label, y_label = "x (a₀)", "y (a₀)"
        elif plane == 'yz':
            Y, Z_grid = xp.meshgrid(g, g, indexing='ij')
            X = xp.full_like(Y, fill_value=position)
            x_label, y_label = "y (a₀)", "z (a₀)"
        else:       
            X, Z_grid = xp.meshgrid(g, g, indexing='ij')
            Y = xp.full_like(X, fill_value=position)
            x_label, y_label = "x (a₀)", "z (a₀)"

        R = xp.sqrt(X**2 + Y**2 + Z_grid**2)
        R_safe = xp.where(R == 0, 1e-9, R)
        Theta = xp.arccos(xp.clip(Z_grid / R_safe, -1.0, 1.0))
        Phi = xp.arctan2(Y, X)

        psi = psi_nlm(R, Theta, Phi, n, l, m, Z, real=real)
        prob_density = xp.abs(psi)**2

        if hasattr(prob_density, 'get'):
            prob_cpu, g_cpu = prob_density.get(), g.get()
        else:
            prob_cpu, g_cpu = prob_density, g

        return jsonify({
            "z_data": prob_cpu.T.tolist(),                               
            "x_coords": g_cpu.tolist(), "y_coords": g_cpu.tolist(),
            "x_label": x_label, "y_label": y_label
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route('/api/angular-node-map', methods=['GET'])
def get_angular_node_map():
    """(NEW) Replaces plot_angular_nodes()"""
    try:
        l = int(request.args.get('l'))
        m = int(request.args.get('m'))
        res = int(request.args.get('res'))
        real = request.args.get('real', 'true').lower() == 'true'

        phi = xp.linspace(0, 2 * np.pi, res, dtype=xp.float32)
        theta = xp.linspace(0, np.pi, res, dtype=xp.float32)
        phi_grid, theta_grid = xp.meshgrid(phi, theta)

        Y_vals = Y_lm(l, m, theta_grid, phi_grid, real=real)
        prob_density = xp.abs(Y_vals)**2

        if hasattr(prob_density, 'get'):
            prob_cpu, phi_cpu, theta_cpu = prob_density.get(), phi.get(), theta.get()
        else:
            prob_cpu, phi_cpu, theta_cpu = prob_density, phi, theta

        return jsonify({
            "z_data": prob_cpu.T.tolist(),                       
            "x_coords": phi_cpu.tolist(),
            "y_coords": theta_cpu.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route('/api/total-isosurface', methods=['GET'])
def get_total_isosurface():
    """(REWRITTEN) Calculates shape from log(|Psi|^2) but color from phase(Psi)"""
    if not SKIMAGE_AVAILABLE:
        return jsonify({"error": "scikit-image library not found on server."}), 500

    try:
        n = int(request.args.get('n'))
        l = int(request.args.get('l'))
        m = int(request.args.get('m'))
        Z = int(request.args.get('Z'))
        res = int(request.args.get('res'))
        r_max = float(request.args.get('r_max'))
        real = request.args.get('real', 'true').lower() == 'true'

        _, _, _, R, Theta, Phi, g = get_3d_grid_coords(res, r_max)
        psi = psi_nlm(R, Theta, Phi, n, l, m, Z, real=real)

        if real:
                                                                 
            color_grid = xp.sign(psi + 1e-15 * (xp.random.rand(*psi.shape) - 0.5))
            colorscale = 'RdBu'
            cmin, cmax = -1.0, 1.0
            colorbar_title = 'Sign (+/-)'
        else:
            color_grid = xp.angle(psi + 1e-15)              
            colorscale = 'hsv'
            cmin, cmax = -np.pi, np.pi
            colorbar_title = 'Phase (rad)'

        prob_density = xp.abs(psi)**2
        max_prob = xp.max(prob_density)
        if max_prob <= 1e-12:
            return jsonify({"error": "Max probability is effectively zero."}), 500

        floor_prob = max_prob * 1e-7
        prob_density_clipped = xp.maximum(prob_density, floor_prob)
        surface_grid = xp.log10(prob_density_clipped)

        surf_min, surf_max = float(xp.min(surface_grid)), float(xp.max(surface_grid))

        meshes = compute_3d_isosurfaces(surface_grid, color_grid, surf_min, surf_max, g)

        return jsonify({
            "meshes": meshes,
            "cmin": cmin, "cmax": cmax,
            "colorscale": colorscale, "colorbar_title": colorbar_title
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route('/api/energy-levels', methods=['GET'])
def get_energy_levels():
    """(MODIFIED) Now includes transition series"""
    try:
        Z = int(request.args.get('Z'))
        n_max = int(request.args.get('n_max'))

        levels = []
        for n in range(1, n_max + 1):
            _, E_eV = calculate_energy(n, Z)
            for l in range(n):
                levels.append({
                    "n": n, "l": l, "E_eV": E_eV,
                    "label": f"{n}{ORBITAL_LABELS.get(l, '?')}"
                })

        transitions = []
        series = {
            "Lyman": 1, "Balmer": 2, "Paschen": 3, "Brackett": 4, "Pfund": 5
        }
        for name, n_final in series.items():
            if n_final >= n_max: continue                                          
                                     
            final_levels = [lvl for lvl in levels if lvl['n'] == n_final]
            if not final_levels: continue

            initial_levels = [lvl for lvl in levels if lvl['n'] > n_final]

            for i_lvl in initial_levels:
                for f_lvl in final_levels:
                                             
                    if abs(i_lvl['l'] - f_lvl['l']) == 1:
                        transitions.append({
                            "x_start": i_lvl['l'], "y_start": i_lvl['E_eV'],
                            "x_end": f_lvl['l'], "y_end": f_lvl['E_eV'],
                            "series": name
                        })

        return jsonify({"levels": levels, "transitions": transitions})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route('/api/compare-orbitals', methods=['POST'])
def get_orbital_comparison():
    """(NEW) Replaces compare_orbitals()"""
    try:
        request_data = request.json
        if request_data is None:
            return jsonify({"error": "No JSON data received"}), 400
        orbitals = request_data.get('orbitals', [])                    
        Z = int(request_data.get('Z', 1))
        r_max = float(request_data.get('r_max', 20))
        res = int(request_data.get('res', 100))             

        g = xp.linspace(-r_max, r_max, res, dtype=xp.float32)
        X, Z_grid = xp.meshgrid(g, g, indexing='ij')
        Y = xp.zeros_like(X)

        R = xp.sqrt(X**2 + Y**2 + Z_grid**2)
        R_safe = xp.where(R == 0, 1e-9, R)
        Theta = xp.arccos(xp.clip(Z_grid / R_safe, -1.0, 1.0))
        Phi = xp.arctan2(Y, X)

        if hasattr(g, 'get'):
            g_cpu = g.get()
        else:
            g_cpu = g

        max_z = 0
        plots = []

        for (n, l, m) in orbitals:
            psi = psi_nlm(R, Theta, Phi, n, l, m, Z, real=True)
            prob_density = xp.abs(psi)**2
            if hasattr(prob_density, 'get'):
                prob_cpu = prob_density.get()
            else:
                prob_cpu = prob_density

            plots.append({
                "z_data": prob_cpu.T.tolist(),
                "title": f"n={n}, l={l}, m={m}"
            })
            max_z = max(max_z, prob_cpu.max() if prob_cpu.size > 0 else 0)

        return jsonify({
            "plots": plots,
            "x_coords": g_cpu.tolist(),
            "y_coords": g_cpu.tolist(),
            "max_z": float(max_z) if max_z > 1e-12 else 1.0                         
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route('/api/superposition-animation', methods=['POST'])
def get_superposition_animation():
    """(MODIFIED) Replaces create_superposition_animation, now with selectable plane"""
    try:
        request_data = request.json
        states = request_data.get('states', [])                                
        Z = int(request_data.get('Z', 1))
        r_max = float(request_data.get('r_max', 20))
        res = int(request_data.get('res', 100))         
        num_frames = int(request_data.get('num_frames', 50))
        plane = request_data.get('plane', 'xz')                    

        g = xp.linspace(-r_max, r_max, res, dtype=xp.float32)

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

        R = xp.sqrt(X**2 + Y**2 + Z_grid**2)
        R_safe = xp.where(R == 0, 1e-9, R)
        Theta = xp.arccos(xp.clip(Z_grid / R_safe, -1.0, 1.0))
        Phi = xp.arctan2(Y, X)

        psi_states, energies, coeffs = [], [], []

        for (n, l, m, real_c, imag_c) in states:
            psi = psi_nlm(R, Theta, Phi, n, l, m, Z, real=False)
            psi_states.append(psi)

            E_hartree_bohr, E_eV_bohr = calculate_energy(n, Z)
            dE_rel_eV = relativistic_correction(n, l, Z)
            dE_dar_eV = darwin_correction(n, l, Z)
            E_total_eV = E_eV_bohr + dE_rel_eV + dE_dar_eV
            E_total_hartree = E_total_eV / HARTREE_TO_EV
            energies.append(E_total_hartree)

            coeffs.append(complex(real_c, imag_c))

        coeffs = np.array(coeffs, dtype=np.complex128)
        norm = np.sqrt(np.sum(np.abs(coeffs)**2))
        if norm > 1e-9:
             coeffs = coeffs / norm
        else:
             return jsonify({"error": "Coefficients cannot all be zero."}), 400

        energies_np = np.array(energies)
        E_diffs = np.abs(energies_np - energies_np[:, None])
        non_zero_diffs = E_diffs[E_diffs > 1e-12]
        if len(non_zero_diffs) == 0:
             min_E_diff = 1.0
        else:
             min_E_diff = np.min(non_zero_diffs)
        T_period = 2 * np.pi / min_E_diff

        times = xp.linspace(0, T_period, num_frames)

        z_max = 0
        frame_data = []
        for t in times:
            psi_t = xp.zeros_like(psi_states[0], dtype=xp.complex128)
            for i, (psi_i, E_i, c_i) in enumerate(zip(psi_states, energies, coeffs)):
                phase = xp.exp(-1j * E_i * t)
                psi_t += c_i * psi_i * phase

            prob_density = xp.abs(psi_t)**2
            if hasattr(prob_density, 'get'):
                prob_cpu = prob_density.get()
            else:
                prob_cpu = prob_density

            frame_data.append(prob_cpu.T.tolist())
            z_max = max(z_max, prob_cpu.max() if prob_cpu.size > 0 else 0)

        if hasattr(g, 'get'):
            g_cpu, times_cpu = g.get(), times.get()
        else:
            g_cpu, times_cpu = g, times

        return jsonify({
            "frames": frame_data,
            "times": times_cpu.tolist(),
            "x_coords": g_cpu.tolist(), "y_coords": g_cpu.tolist(),
            "x_label": x_label, "y_label": y_label,                   
            "z_max": float(z_max) if z_max > 1e-12 else 1.0,
            "period": float(T_period)
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500    

@app.route('/api/splitting-diagram', methods=['GET'])
def get_splitting_diagram():
    """(NEW) Replaces plot_correction_comparison()"""
    try:
        n = int(request.args.get('n'))
        Z = int(request.args.get('Z'))

        categories = [
            'Bohr (n)', 'Bohr + Relativistic (n,l)', 'Bohr + Rel + Darwin (n,l)',
            'Full Fine Structure (n,l,j)', 'Full FS + Hyperfine (n,l,j,F)'
        ]
        traces = []

        for l in range(n):
            _, E_Bohr_val = calculate_energy(n, Z)
            dE_Rel = relativistic_correction(n, l, Z)
            E_Rel_val = E_Bohr_val + dE_Rel
            dE_Dar = darwin_correction(n, l, Z)
            E_Dar_val = E_Rel_val + dE_Dar

            j_states = [0.5] if l == 0 else sorted([l - 0.5, l + 0.5])

            for j in j_states:
                                                                                       
                dE_FS_total_rel_bohr = calculate_fine_structure(n, l, j, Z)
                E_FS_val = E_Bohr_val + dE_FS_total_rel_bohr

                F_states = [j + I_proton]
                                                                          
                if (j - I_proton >= -1e-9) and not np.isclose(j - I_proton, j + I_proton):
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

                    traces.append({
                        "x": x_data, "y": y_data,
                        "mode": 'lines+markers', "name": state_label,
                        "hovertemplate": f"<b>{state_label}</b><br>Level: %{{x}}<br>Energy: %{{y:.12f}} eV<extra></extra>"
                    })

        return jsonify({"traces": traces, "categories": categories})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  Hydrogen Atom Real-Time Calculation Server (v5.2 - Final)")
    print("="*60)
    print(f"  Backend: {'CuPy (GPU)' if xp == cp else 'NumPy (CPU)'}")
    print(f"  Isosurface (scikit-image): {'Available' if SKIMAGE_AVAILABLE else 'NOT FOUND'}")
    print(f"  3D Interpolation (scipy.ndimage): {'Available' if SCIPY_NDIMAGE_AVAILABLE else 'NOT FOUND'}")
    print(f"\n  Server running at http://127.0.0.1:5000")
    print("  Keep this terminal running.")
    print("  Open 'index.html' in your browser to use the application.")
    app.run(host='0.0.0.0', port=5000)