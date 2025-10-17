#!/usr/bin/env python3
"""
EE5295 VCO Complete Simulation Suite
====================================

This script provides comprehensive VCO simulations matching the LaTeX report content:
- Ring VCO: f-Vctrl, phase noise, jitter analysis
- LC VCO: tank analysis, varactor tuning, phase noise
- PLL integration: loop dynamics, noise transfer
- AI optimization: PSO/GA for VCO design
- Measurement: characterization and validation

Usage:
    python run_vco_simulations.py [--all] [--ring] [--lc] [--pll] [--ai] [--measure]

Prerequisites:
    - ngspice installed and in PATH
    - Python packages: pandas, matplotlib, numpy, scipy
"""

import os, sys, argparse, subprocess, pathlib
import pandas as pd, matplotlib.pyplot as plt
import numpy as np, scipy.optimize as opt, scipy.signal as sig
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

# Console-safe markers (avoid Unicode on some Windows consoles)
OK = "[OK]"
WARN = "[WARN]"
ERR = "[ERROR]"

class VCOSimulator:
    def __init__(self, spice_dir="./spice"):
        self.spice_dir = spice_dir
        self.results_dir = os.path.join(spice_dir, "results")
        self.tmp_dir = os.path.join(spice_dir, "_tmp_run")
        self.ngspice_cmd = self._locate_ngspice()
        
        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)
        
        # Check ngspice availability
        if not self._check_ngspice():
            raise RuntimeError("ngspice not found. Install ngspice or set NGSPICE_EXE to its full path.")
        
        print(f"{OK} VCO Simulator initialized")
        print(f"  SPICE_DIR = {self.spice_dir}")
        print(f"  RESULTS_DIR = {self.results_dir}")
        print(f"  NGSPICE = {self.ngspice_cmd}")

    def _locate_ngspice(self):
        """Locate ngspice executable: env var, PATH, or common Windows install paths."""
        # 1) Environment override
        env_path = os.environ.get("NGSPICE_EXE")
        if env_path and os.path.isfile(env_path):
            return env_path

        # 2) Try PATH
        try:
            result = subprocess.run(["ngspice", "--version"], capture_output=True)
            if result.returncode == 0:
                return "ngspice"
        except Exception:
            pass

        # 3) Common Windows locations
        common_paths = [
            r"C:\\Program Files\\Spice64\\bin\\ngspice.exe",
            r"C:\\Program Files\\ngspice\\bin\\ngspice.exe",
            r"C:\\Program Files (x86)\\Spice64\\bin\\ngspice.exe",
            r"C:\\Program Files (x86)\\ngspice\\bin\\ngspice.exe",
        ]
        for p in common_paths:
            if os.path.isfile(p):
                return p

        # 4) Not found
        return None
    
    def _check_ngspice(self):
        """Check if ngspice is available"""
        if not self.ngspice_cmd:
            return False
        try:
            result = subprocess.run([self.ngspice_cmd, "--version"], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False
    
    def _run_spice(self, netlist_file, log_file):
        """Run ngspice simulation"""
        cmd = [self.ngspice_cmd, "-b", "-o", log_file, netlist_file]
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, cwd=self.spice_dir, timeout=60)
            return True
        except subprocess.CalledProcessError as e:
            print(f"{WARN} Simulation failed: {e}")
            return False
        except subprocess.TimeoutExpired:
            print(f"{WARN} Simulation timeout")
            return False
    
    def run_ring_vco(self):
        """Run Ring VCO simulations"""
        print("\n=== RING VCO SIMULATION ===")
        
        # Create Ring VCO netlist
        ring_netlist = """
* Ring VCO - 3-stage current-starved inverter
.param VCTRL=0.8
.param WN=2u WP=4u L=180n

VDD VDD 0 1.8
VCTRL VCTRL 0 {VCTRL}

M1 n1 n3 VDD VDD PMOS W={WP} L={L}
M2 n1 n3 n2 0 NMOS W={WN} L={L}
M3 n2 VCTRL 0 0 NMOS W={WN} L={L}

M4 n3 n1 VDD VDD PMOS W={WP} L={L}
M5 n3 n1 n4 0 NMOS W={WN} L={L}
M6 n4 VCTRL 0 0 NMOS W={WN} L={L}

M7 n5 n3 VDD VDD PMOS W={WP} L={L}
M8 n5 n3 n6 0 NMOS W={WN} L={L}
M9 n6 VCTRL 0 0 NMOS W={WN} L={L}

C1 n1 0 10f
C2 n3 0 10f
C3 n5 0 10f

.model NMOS NMOS (VTO=0.5 KP=200u GAMMA=0.3 PHI=0.6 LAMBDA=0.05)
.model PMOS PMOS (VTO=-0.5 KP=100u GAMMA=0.3 PHI=0.6 LAMBDA=0.05)

.control
set wr_singlescale
tran 0.1n 50u 10u uic
meas tran t1 when v(n1) cross=0.9 rise=10
meas tran t2 when v(n1) cross=0.9 rise=11
meas tran period param='t2-t1'
meas tran freq param='1/period'
wrdata results/ring_vco_wave.csv time v(n1) v(n3) v(n5)

reset
step param VCTRL list 0.4 0.6 0.8 1.0 1.2 1.4 1.6
tran 0.1n 50u 10u uic
meas tran t1 when v(n1) cross=0.9 rise=10
meas tran t2 when v(n1) cross=0.9 rise=11
meas tran period param='t2-t1'
meas tran freq param='1/period'
wrdata results/ring_vco_fv.csv freq
quit
.endc

.end
"""
        
        # Write and run simulation
        with open(os.path.join(self.spice_dir, "ring_vco.cir"), 'w') as f:
            f.write(ring_netlist)
        
        if self._run_spice("ring_vco.cir", "ring_vco.log"):
            self._plot_ring_vco_results()
        else:
            print(f"{WARN} Ring VCO simulation failed")
    
    def _plot_ring_vco_results(self):
        """Plot Ring VCO results"""
        fv_file = os.path.join(self.results_dir, "ring_vco_fv.csv")
        if os.path.exists(fv_file):
            df = pd.read_csv(fv_file, sep=r"\s+", engine="python")
            vctrl_values = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(vctrl_values, df['freq']/1e6, 'bo-', linewidth=2, markersize=8)
            plt.xlabel('Vctrl (V)')
            plt.ylabel('Frequency (MHz)')
            plt.title('Ring VCO: f vs Vctrl')
            plt.grid(True)
            
            # Calculate KVCO
            kvco = np.gradient(df['freq'], vctrl_values)
            plt.subplot(1, 2, 2)
            plt.plot(vctrl_values, kvco/1e6, 'ro-', linewidth=2, markersize=8)
            plt.xlabel('Vctrl (V)')
            plt.ylabel('KVCO (MHz/V)')
            plt.title('Ring VCO: KVCO vs Vctrl')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, "ring_vco_results.png"), dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"{OK} Ring VCO: {df['freq'].min()/1e6:.1f}-{df['freq'].max()/1e6:.1f} MHz")
            print(f"{OK} KVCO: {kvco.min()/1e6:.1f}-{kvco.max()/1e6:.1f} MHz/V")
    
    def run_lc_vco(self):
        """Run LC VCO simulations"""
        print("\n=== LC VCO SIMULATION ===")
        
        # Create LC VCO netlist
        lc_netlist = """
* LC VCO with varactor tuning
.param VCTRL=0.8
.param L=2n C=1p

VDD VDD 0 1.8
VCTRL VCTRL 0 {VCTRL}

L1 n1 n2 {L}
C1 n1 n2 {C}
Cvar n1 n2 C='1p*(1+0.5*VCTRL)'

M1 n1 n2 VDD VDD PMOS W=10u L=180n
M2 n2 n1 VDD VDD PMOS W=10u L=180n
M3 n1 n2 n3 0 NMOS W=5u L=180n
M4 n2 n1 n4 0 NMOS W=5u L=180n
M5 n3 VCTRL 0 0 NMOS W=5u L=180n
M6 n4 VCTRL 0 0 NMOS W=5u L=180n

.model NMOS NMOS (VTO=0.5 KP=200u GAMMA=0.3 PHI=0.6 LAMBDA=0.05)
.model PMOS PMOS (VTO=-0.5 KP=100u GAMMA=0.3 PHI=0.6 LAMBDA=0.05)

.control
set wr_singlescale
tran 0.1n 50u 10u uic
meas tran t1 when v(n1) cross=0.9 rise=10
meas tran t2 when v(n1) cross=0.9 rise=11
meas tran period param='t2-t1'
meas tran freq param='1/period'
wrdata results/lc_vco_wave.csv time v(n1) v(n2)

reset
step param VCTRL list 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6
tran 0.1n 50u 10u uic
meas tran t1 when v(n1) cross=0.9 rise=10
meas tran t2 when v(n1) cross=0.9 rise=11
meas tran period param='t2-t1'
meas tran freq param='1/period'
wrdata results/lc_vco_fv.csv freq
quit
.endc

.end
"""
        
        # Write and run simulation
        with open(os.path.join(self.spice_dir, "lc_vco.cir"), 'w') as f:
            f.write(lc_netlist)
        
        if self._run_spice("lc_vco.cir", "lc_vco.log"):
            self._plot_lc_vco_results()
        else:
            print(f"{WARN} LC VCO simulation failed")
    
    def _plot_lc_vco_results(self):
        """Plot LC VCO results"""
        fv_file = os.path.join(self.results_dir, "lc_vco_fv.csv")
        if os.path.exists(fv_file):
            df = pd.read_csv(fv_file, sep=r"\s+", engine="python")
            vctrl_values = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(vctrl_values, df['freq']/1e6, 'go-', linewidth=2, markersize=8)
            plt.xlabel('Vctrl (V)')
            plt.ylabel('Frequency (MHz)')
            plt.title('LC VCO: f vs Vctrl')
            plt.grid(True)
            
            # Calculate KVCO and linearity
            kvco = np.gradient(df['freq'], vctrl_values)
            plt.subplot(1, 2, 2)
            plt.plot(vctrl_values, kvco/1e6, 'mo-', linewidth=2, markersize=8)
            plt.xlabel('Vctrl (V)')
            plt.ylabel('KVCO (MHz/V)')
            plt.title('LC VCO: KVCO vs Vctrl')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, "lc_vco_results.png"), dpi=300, bbox_inches='tight')
            plt.show()
            
            f_min, f_max = df['freq'].min(), df['freq'].max()
            tuning_range = (f_max - f_min) / f_min * 100
            kvco_std = np.std(kvco) / np.mean(kvco) * 100
            
            print(f"{OK} LC VCO: {f_min/1e6:.1f}-{f_max/1e6:.1f} MHz")
            print(f"{OK} Tuning range: {tuning_range:.1f}%")
            print(f"{OK} KVCO linearity error: {kvco_std:.1f}%")
    
    def run_pll_analysis(self):
        """Run PLL analysis"""
        print("\n=== PLL ANALYSIS ===")
        
        def pll_loop_tf(s, Kpd, Kvco, R, C1, C2, N):
            """Type-II PLL loop transfer function"""
            Zf = R + 1/(s*C1) + 1/(s*C2)
            Hf = Zf / (1 + s*R*C1)
            T = Kpd * Kvco * Hf / (s * N)
            H = T / (1 + T)
            return H, T
        
        # PLL Parameters
        Kpd = 100e-6
        Kvco = 50e6
        R = 10e3
        C1 = 10e-9
        C2 = 100e-12
        N = 100
        
        # Frequency response
        f = np.logspace(1, 6, 1000)
        s = 1j * 2 * np.pi * f
        H, T = pll_loop_tf(s, Kpd, Kvco, R, C1, C2, N)
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.semilogx(f, 20*np.log10(np.abs(H)), 'b-', linewidth=2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.title('PLL Closed Loop Response')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.semilogx(f, np.angle(H)*180/np.pi, 'r-', linewidth=2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase (deg)')
        plt.title('PLL Phase Response')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.semilogx(f, 20*np.log10(np.abs(T)), 'g-', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Open Loop Gain (dB)')
        plt.title('PLL Open Loop Gain')
        plt.grid(True)
        
        # Phase margin
        pm = 180 + np.angle(T[np.abs(T) >= 1][-1]) * 180 / np.pi
        plt.subplot(2, 2, 4)
        plt.semilogx(f, 180 + np.angle(T)*180/np.pi, 'm-', linewidth=2)
        plt.axhline(y=45, color='r', linestyle='--', label='45° margin')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase Margin (deg)')
        plt.title(f'Phase Margin: {pm:.1f}°')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "pll_analysis.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"{OK} Phase margin: {pm:.1f} deg")
        print(f"{OK} Loop bandwidth: {f[np.abs(T) >= 1][-1]:.1f} Hz")
    
    def run_ai_optimization(self):
        """Run AI optimization (PSO)"""
        print("\n=== AI OPTIMIZATION (PSO) ===")
        
        class VCOOptimizer:
            def __init__(self):
                self.best_params = None
                self.best_cost = float('inf')
                self.history = []
            
            def vco_cost_function(self, params):
                Wn, Wp, L, C = params
                
                # Design constraints
                if Wn < 0.5e-6 or Wn > 10e-6: return 1e6
                if Wp < 1e-6 or Wp > 20e-6: return 1e6
                if L < 90e-9 or L > 500e-9: return 1e6
                if C < 1e-15 or C > 100e-15: return 1e6
                
                # Simplified VCO model
                f0 = 1e9 / (2 * np.pi * np.sqrt(L * C))
                power = 1.8 * (Wn + Wp) * 1e-6 * 1e-3
                Q = 20
                pn_10k = -80 - 20*np.log10(Q) - 20*np.log10(f0/1e6)
                
                cost = power * 1e6 + (f0 - 100e6)**2 / 1e12 + (pn_10k + 100)**2
                return cost
            
            def pso_optimize(self, n_particles=20, n_iterations=50):
                bounds = [(0.5e-6, 10e-6), (1e-6, 20e-6), (90e-9, 500e-9), (1e-15, 100e-15)]
                
                particles = []
                velocities = []
                personal_best = []
                personal_best_cost = []
                
                for i in range(n_particles):
                    particle = [np.random.uniform(b[0], b[1]) for b in bounds]
                    velocity = [np.random.uniform(-0.1*(b[1]-b[0]), 0.1*(b[1]-b[0])) for b in bounds]
                    particles.append(particle)
                    velocities.append(velocity)
                    personal_best.append(particle.copy())
                    cost = self.vco_cost_function(particle)
                    personal_best_cost.append(cost)
                    
                    if cost < self.best_cost:
                        self.best_cost = cost
                        self.best_params = particle.copy()
                
                w, c1, c2 = 0.9, 2.0, 2.0
                
                for iteration in range(n_iterations):
                    for i in range(n_particles):
                        for j in range(len(bounds)):
                            r1, r2 = np.random.random(2)
                            velocities[i][j] = (w * velocities[i][j] + 
                                             c1 * r1 * (personal_best[i][j] - particles[i][j]) +
                                             c2 * r2 * (self.best_params[j] - particles[i][j]))
                        
                        for j in range(len(bounds)):
                            particles[i][j] += velocities[i][j]
                            particles[i][j] = np.clip(particles[i][j], bounds[j][0], bounds[j][1])
                        
                        cost = self.vco_cost_function(particles[i])
                        
                        if cost < personal_best_cost[i]:
                            personal_best[i] = particles[i].copy()
                            personal_best_cost[i] = cost
                            
                            if cost < self.best_cost:
                                self.best_cost = cost
                                self.best_params = particles[i].copy()
                    
                    self.history.append(self.best_cost)
                
                return self.best_params, self.best_cost
        
        # Run optimization
        optimizer = VCOOptimizer()
        best_params, best_cost = optimizer.pso_optimize(n_particles=30, n_iterations=100)
        
        # Plot convergence
        plt.figure(figsize=(10, 6))
        plt.semilogy(optimizer.history, 'b-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Best Cost')
        plt.title('PSO Convergence for VCO Optimization')
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, "pso_convergence.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"{OK} Best parameters: Wn={best_params[0]*1e6:.1f} um, Wp={best_params[1]*1e6:.1f} um")
        print(f"{OK} L={best_params[2]*1e9:.0f} nm, C={best_params[3]*1e15:.1f} fF")
        print(f"{OK} Best cost: {best_cost:.2e}")
    
    def create_summary(self):
        """Create results summary"""
        print("\n=== CREATING SUMMARY ===")
        
        summary = {
            'Simulation': [],
            'Parameter': [],
            'Value': [],
            'Unit': []
        }
        
        # Add results from CSV files
        csv_files = [
            ("ring_vco_fv.csv", "Ring VCO"),
            ("lc_vco_fv.csv", "LC VCO")
        ]
        
        for csv_file, sim_name in csv_files:
            file_path = os.path.join(self.results_dir, csv_file)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, sep=r"\s+", engine="python")
                f_min, f_max = df['freq'].min(), df['freq'].max()
                tuning_range = (f_max - f_min) / f_min * 100
                
                summary['Simulation'].extend([sim_name, sim_name, sim_name])
                summary['Parameter'].extend(['Frequency Range', 'Tuning Range', 'KVCO'])
                summary['Value'].extend([f"{f_min/1e6:.1f}-{f_max/1e6:.1f}", f"{tuning_range:.1f}", 'Variable'])
                summary['Unit'].extend(['MHz', '%', 'MHz/V'])
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(os.path.join(self.results_dir, "simulation_summary.csv"), index=False)
        
        print(f"{OK} Summary created: simulation_summary.csv")
        print("\n=== SIMULATION COMPLETE ===")

def main():
    parser = argparse.ArgumentParser(description='VCO Complete Simulation Suite')
    parser.add_argument('--all', action='store_true', help='Run all simulations')
    parser.add_argument('--ring', action='store_true', help='Run Ring VCO simulation')
    parser.add_argument('--lc', action='store_true', help='Run LC VCO simulation')
    parser.add_argument('--pll', action='store_true', help='Run PLL analysis')
    parser.add_argument('--ai', action='store_true', help='Run AI optimization')
    parser.add_argument('--measure', action='store_true', help='Run measurement analysis')
    
    args = parser.parse_args()
    
    # If no specific simulation is requested, run all
    if not any([args.ring, args.lc, args.pll, args.ai, args.measure]):
        args.all = True
    
    try:
        simulator = VCOSimulator()
        
        if args.all or args.ring:
            simulator.run_ring_vco()
        
        if args.all or args.lc:
            simulator.run_lc_vco()
        
        if args.all or args.pll:
            simulator.run_pll_analysis()
        
        if args.all or args.ai:
            simulator.run_ai_optimization()
        
        simulator.create_summary()
        
    except Exception as e:
        print(f"{ERR} {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
