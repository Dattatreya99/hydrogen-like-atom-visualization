# Hydrogen-Like Atom Orbital Visualization Toolkit

A comprehensive Python-based visualization suite for exploring hydrogen-like atomic orbitals with quantum mechanical accuracy. This toolkit provides both interactive 3D visualizations and detailed physical analysis, including fine structure, hyperfine splitting, and time-dependent superposition states.

## Features

### Core Visualizations
- **2D Radial Distribution**: Plot radial probability density P(r) = r²|R(r)|²
- **3D Volumetric Rendering**: Phase-colored isosurface "clouds" showing probability distributions
- **Angular Distribution**: Spherical harmonic visualization with phase coloring
- **Planar Slices**: 2D cross-sections through the wavefunction
- **Angular Node Maps**: Visualization of nodal surfaces in (θ, φ) space

### Advanced Features
- **Energy Level Diagrams**: Bohr model energy levels with spectral series transitions
- **Fine Structure Corrections**: Relativistic, Darwin term, and spin-orbit coupling
- **Hyperfine Splitting**: Fermi contact term for hydrogen ground state
- **Orbital Comparison**: Side-by-side visualization of multiple orbitals
- **Time-Dependent Superposition**: Animated evolution of quantum superposition states
- **Wavefunction Normalization Verification**: Monte Carlo integration checks

### Physical Accuracy
- Exact analytical hydrogen-like wavefunctions
- Support for arbitrary atomic number Z (H, He⁺, Li²⁺, etc.)
- Quantum corrections: relativistic, Darwin, spin-orbit, hyperfine
- Proper normalization and expectation value calculations
- Radial node detection using Laguerre polynomial roots

## Installation

### Prerequisites
```bash
# Core requirements
pip install numpy scipy plotly flask flask-cors scikit-image matplotlib

# Optional: GPU acceleration (NVIDIA CUDA required)
pip install cupy-cuda12x  # Replace with your CUDA version

# Optional: Progress bars
pip install tqdm
```

### Clone Repository
```bash
git clone https://github.com/yourusername/hydrogen-orbital-viz.git
cd hydrogen-orbital-viz
```

## Usage

### Option 1: Interactive CLI (Python)

Run the standalone Python script for terminal-based interaction:

```bash
python Hydrogen_like_atom.py
```

**Example workflow:**
1. Enter quantum numbers (Z, n, l, m)
2. Choose visualization quality (Low/Medium/High/Ultra)
3. Select from menu options:
   - Plot radial distributions
   - Generate 3D visualizations
   - Compare orbitals
   - Create animations

**Sample inputs:**
```
Atomic number Z: 1        # Hydrogen
Principal n: 3            # Third shell
Angular l: 2              # d-orbital
Magnetic m: 1             # m_l = 1
```

### Option 2: Web Interface (Recommended)

Launch the Flask server and use the interactive web interface:

```bash
# Terminal 1: Start the calculation server
python server.py

# Terminal 2: Open index.html in your browser
# Or simply double-click index.html
```

**Web interface features:**
- Real-time parameter adjustment
- Interactive 3D plots (rotate, zoom, pan)
- Multiple visualization modes accessible via buttons
- Automatically synchronized controls

## Architecture

```
├── Hydrogen_like_atom.py    # Core physics library (standalone + CLI)
├── server.py                # Flask API server for web interface
├── index.html               # Web-based visualization frontend
└── README.md                # This file
```

### Component Breakdown

**Hydrogen_like_atom.py**
- Wavefunction calculations (radial R_nl, angular Y_lm, total ψ_nlm)
- Quantum corrections and energy calculations
- Plotting functions using Plotly
- GPU acceleration support via CuPy

**server.py**
- Flask REST API endpoints for each visualization
- 3D isosurface generation using marching cubes
- Data serialization for web transfer
- Error handling and validation

**index.html**
- Dark-themed responsive UI
- Real-time plot generation via Plotly.js
- Parameter validation and control synchronization
- Animation playback controls

## Quantum Numbers Reference

| Symbol | Name | Range | Description |
|--------|------|-------|-------------|
| **Z** | Atomic number | Z ≥ 1 | Nuclear charge (1=H, 2=He⁺, etc.) |
| **n** | Principal | n ≥ 1 | Energy level, shell |
| **l** | Angular momentum | 0 ≤ l < n | Orbital shape (0=s, 1=p, 2=d, 3=f) |
| **m** | Magnetic | -l ≤ m ≤ l | Orbital orientation |
| **j** | Total angular | \|l - 1/2\| or l + 1/2 | Fine structure splitting |
| **F** | Hyperfine | j ± I | Hyperfine structure (H only) |

## Examples

### Example 1: Hydrogen 3d Orbital
```python
# Via CLI
Z=1, n=3, l=2, m=0  # 3d_z² orbital

# Via web interface
# Set controls: Z=1, n=3, l=2, m=0
# Click "Plot Total |Ψ| (3D)"
```

### Example 2: Helium Ion Ground State
```python
Z=2, n=1, l=0, m=0  # He⁺ 1s orbital
# Visualize increased nuclear attraction (tighter wavefunction)
```

### Example 3: Superposition Animation
```python
# Combine 2p states to show orbital precession
States: (2,1,1) + (2,1,-1) with equal coefficients
# Generates rotating "dumbbell" animation
```

## Performance Notes

- **Low quality** (40³ grid): Fast, suitable for exploration
- **Medium quality** (60³ grid): Balanced performance (default)
- **High quality** (80³ grid): Detailed features, slower rendering
- **Ultra quality** (100³ grid): Publication-quality, GPU recommended

**GPU Acceleration:**
- Automatically detected if CuPy is installed
- 5-10× speedup for large 3D grids
- Required CUDA-capable NVIDIA GPU

## Physics Background

### Energy Levels
**Bohr model:**
```
E_n = -Z²/(2n²) Hartree = -13.6 Z²/n² eV
```

**Fine structure correction:**
```
ΔE_fs = (E_n × (Zα)²/n²) × [n/(j+1/2) - 3/4]
```

**Hyperfine splitting (l=0, Z=1 only):**
```
ΔE_hfs ≈ 5.88 × 10⁻⁶ eV  (1420 MHz, 21-cm line)
```

### Wavefunctions
**Radial part:**
```
R_nl(r) = N × exp(-ρ/2) × ρ^l × L_{n-l-1}^{2l+1}(ρ)
where ρ = 2Zr/(na₀)
```

**Angular part (spherical harmonics):**
```
Y_l^m(θ,φ) = N × P_l^{|m|}(cos θ) × exp(imφ)
```

## Troubleshooting

**Issue: "scikit-image not found"**
```bash
pip install scikit-image
```

**Issue: Server won't start**
- Check if port 5000 is available
- Try changing the port in `server.py`: `app.run(port=5001)`

**Issue: Plots are too slow**
- Reduce quality setting to "Low" or "Medium"
- Install CuPy for GPU acceleration
- Close other applications consuming GPU/CPU

**Issue: "Invalid quantum numbers"**
- Verify: n > l ≥ 0 and |m| ≤ l
- Example: n=3 allows l=0,1,2; l=1 allows m=-1,0,1

## Contributing

Contributions are welcome! Areas for improvement:
- Additional quantum corrections (Lamb shift, vacuum polarization)
- Support for multi-electron atoms (Hartree-Fock approximations)
- Export to 3D formats (STL, OBJ) for 3D printing
- More spectroscopic tools (transition dipole moments, selection rules)

## License

MIT License - see LICENSE file for details

## Citation

If you use this tool in your research, please cite:
```bibtex
@software{hydrogen_orbital_viz,
  author = {Dattatreya Mangipudi},
  title = {Hydrogen-Like Atom Orbital Visualization Toolkit},
  year = {2025},
  url = {https://github.com/yourusername/hydrogen-orbital-viz}
}
```

## References

1. Griffiths, D. J. (2018). *Introduction to Quantum Mechanics* (3rd ed.)
2. Sakurai, J. J. (2020). *Modern Quantum Mechanics* (3rd ed.)
3. NIST Atomic Spectra Database: https://physics.nist.gov/asd

## Acknowledgments

Built with:
- NumPy/SciPy for numerical computation
- Plotly for interactive visualizations
- Flask for web API
- scikit-image for isosurface generation
- CuPy for GPU acceleration (optional)

---

**Project Status:** Active development  
**Last Updated:** December 2025  
**Python Version:** 3.8+
