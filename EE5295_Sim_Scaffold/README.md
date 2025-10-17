# EE5295 VCO Complete Simulation Suite

This repository provides comprehensive VCO simulations matching the LaTeX report content for EE5295 course.

## Features

- **Ring VCO Analysis**: f-Vctrl characterization, phase noise, jitter analysis
- **LC VCO Analysis**: tank analysis, varactor tuning, phase noise
- **PLL Integration**: loop dynamics, noise transfer functions
- **AI Optimization**: PSO/GA algorithms for VCO design optimization
- **Measurement**: characterization and validation scripts

## Prerequisites

### Software Requirements
- **ngspice**: SPICE circuit simulator
  - Windows: Download from [ngspice.sourceforge.net](http://ngspice.sourceforge.net/)
  - Linux: `sudo apt-get install ngspice`
  - macOS: `brew install ngspice`

- **Python 3.7+** with packages:
  ```bash
  pip install pandas matplotlib numpy scipy scikit-optimize
  ```

### Hardware Requirements
- Minimum 4GB RAM
- 2GB free disk space for simulation results

## Quick Start

### Option 1: Jupyter Notebook (Interactive)
```bash
cd EE5295_Sim_Scaffold
jupyter notebook VCO_Complete_Simulation.ipynb
```

### Option 2: Python Script (Command Line)
```bash
cd EE5295_Sim_Scaffold

# Run all simulations
python run_vco_simulations.py --all

# Run specific simulations
python run_vco_simulations.py --ring --lc --pll
python run_vco_simulations.py --ai
python run_vco_simulations.py --measure
```

## Directory Structure

```
EE5295_Sim_Scaffold/
├── VCO_Complete_Simulation.ipynb    # Main Jupyter notebook
├── run_vco_simulations.py           # Standalone Python script
├── README.md                        # This file
├── spice/                          # SPICE simulation files
│   ├── results/                    # Simulation results (CSV, PNG)
│   └── _tmp_run/                   # Temporary files
└── requirements.txt                 # Python dependencies
```

## Simulation Details

### 1. Ring VCO Analysis
- **3-stage current-starved inverter** topology
- **f-Vctrl characterization** with KVCO calculation
- **Phase noise analysis** (if PSS/Pnoise available)
- **Supply sensitivity** analysis

### 2. LC VCO Analysis
- **Cross-coupled LC tank** with varactor tuning
- **Frequency vs control voltage** characterization
- **KVCO linearity** analysis
- **Tuning range** calculation

### 3. PLL Integration
- **Type-II PLL** loop transfer function
- **Phase margin** and stability analysis
- **Loop bandwidth** calculation
- **Noise transfer** functions

### 4. AI Optimization
- **Particle Swarm Optimization (PSO)** for VCO design
- **Multi-objective optimization**: power, frequency, phase noise
- **Parameter bounds**: transistor sizing, inductor, capacitor
- **Convergence analysis** and visualization

### 5. Measurement & Characterization
- **Supply sensitivity** analysis
- **Temperature coefficient** calculation
- **Jitter analysis** from phase noise
- **Results summary** and export

## Output Files

### Simulation Results
- `ring_vco_fv.csv`: Ring VCO frequency vs control voltage
- `lc_vco_fv.csv`: LC VCO frequency vs control voltage
- `ring_vco_wave.csv`: Ring VCO waveform data
- `lc_vco_wave.csv`: LC VCO waveform data

### Analysis Plots
- `ring_vco_results.png`: Ring VCO f-Vctrl and KVCO plots
- `lc_vco_results.png`: LC VCO f-Vctrl and KVCO plots
- `pll_analysis.png`: PLL transfer function analysis
- `pso_convergence.png`: AI optimization convergence

### Summary
- `simulation_summary.csv`: Comprehensive results summary
- `vco_complete_results.zip`: Complete results archive

## Customization

### Modify VCO Parameters
Edit the netlist parameters in the simulation functions:

```python
# Ring VCO parameters
.param WN=2u WP=4u L=180n

# LC VCO parameters  
.param L=2n C=1p
```

### Adjust AI Optimization
Modify the cost function and bounds in `VCOOptimizer` class:

```python
def vco_cost_function(self, params):
    Wn, Wp, L, C = params
    # Add your custom optimization criteria
    return cost
```

### Change PLL Parameters
Adjust PLL design parameters:

```python
Kpd = 100e-6  # Charge pump current
Kvco = 50e6   # VCO gain
R = 10e3      # Loop filter resistor
C1 = 10e-9    # Loop filter capacitor 1
C2 = 100e-12  # Loop filter capacitor 2
N = 100       # Divider ratio
```

## Troubleshooting

### Common Issues

1. **ngspice not found**
   ```
   Error: ngspice not found in PATH
   ```
   **Solution**: Install ngspice and ensure it's in your system PATH

2. **Phase noise simulation fails**
   ```
   ⚠️ Phase noise simulation failed - PSS/Pnoise may not be available
   ```
   **Solution**: This is normal for some ngspice versions. The simulation will continue with other analyses.

3. **Memory issues with large simulations**
   ```
   ⚠️ Simulation timeout
   ```
   **Solution**: Reduce simulation time or increase system memory

4. **Python package missing**
   ```
   ModuleNotFoundError: No module named 'pandas'
   ```
   **Solution**: Install required packages: `pip install -r requirements.txt`

### Performance Tips

- **Parallel execution**: Run different simulations simultaneously
- **Reduce simulation time**: Modify `tran` parameters for faster execution
- **Memory optimization**: Close unused plots to free memory

## Integration with LaTeX Report

This simulation suite is designed to complement the LaTeX report:

- **Chapter 4**: Ring VCO analysis results
- **Chapter 5**: LC VCO analysis results  
- **Chapter 6**: Phase noise and jitter analysis
- **Chapter 7**: AI optimization methodology
- **Chapter 8**: PLL integration analysis
- **Chapter 9**: Measurement and characterization

## Contributing

To add new simulations or improve existing ones:

1. Fork the repository
2. Create a feature branch
3. Add your simulation code
4. Test thoroughly
5. Submit a pull request

## License

This project is part of EE5295 coursework. Please follow your institution's academic integrity policies.

## Support

For issues or questions:
- Check the troubleshooting section above
- Review ngspice documentation
- Consult the LaTeX report for theoretical background

---

**Note**: This simulation suite is designed for educational purposes and may require adjustments for specific process technologies or design requirements.