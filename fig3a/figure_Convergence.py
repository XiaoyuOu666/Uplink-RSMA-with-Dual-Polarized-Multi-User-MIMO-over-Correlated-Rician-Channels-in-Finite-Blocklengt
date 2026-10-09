"""Generate the fixed-initialization convergence data used in Fig. 3(a)."""

from types import SimpleNamespace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functionUEconfiguration import functionUEconfiguration
from functionSpatialSignature3DLoS import functionSpatialSignature3DLoS
from functionRlocalscattering3D import functionRlocalscattering3D
from functionChannelEstimationDP import functionChannelEstimationDP
from functionChannelEstimationSP import functionChannelEstimationSP
from functionMMSEDetector import functionMMSEDetector
from functionGenerateAV import functionGenerateAV
from functionComputeSEDPnoSIC import functionComputeSEDPnoSIC
from functionComputeSESPSIC import functionComputeSESPSIC
from functionPowerControlDPnoSIC import functionPowerControlDPnoSIC
from functionPowerControlSPSIC import functionPowerControlSPSIC
from functionPowerControlDPnoSIC_Random import functionPowerControlDPnoSIC_Random
from functionPowerControlSPSIC_Random import functionPowerControlSPSIC_Random


def main():
    """Run the simulation and return or export only the data required by the corresponding figure."""
    # Set the common system, reliability, and Monte Carlo parameters.
    K, fc, Nth, Eth = 8, 3.5, 500, 1e-5
    wavelength = 299792458 / (fc * 1e9); noise = 10 ** ((-84 - 30) / 10); rho = Pk = 10 ** ((20 - 30) / 10)
    shadowing = 4 * np.random.randn(K)
    pl1 = lambda d: 10 ** -(3.184 + 2.15 * np.log10(d) + 1.9 * np.log10(fc) + shadowing / 10)
    pl2 = lambda d: 10 ** -(3.363 + 2.19 * np.log10(d) + 2 * np.log10(fc) + shadowing / 10)
    af, ar = 1 / (10 ** 1.5 + 1), 1 / (10 ** .5 + 1)
    # Define the DP and hardware-matched SP array geometries.
    Mv, Mh, M, d, height, scatter = 8, 4, 32, .5, 20, 20
    m = np.arange(M); m2 = np.arange(2 * M)
    posDP = np.vstack((np.zeros(M), m % Mh * d * wavelength, m // Mh * d * wavelength))
    posSP = np.vstack((np.zeros(2 * M), m2 % Mv * d * wavelength, m2 // (2 * Mh) * d * wavelength))
    cfg = SimpleNamespace(InF_width=300, InF_length=300, h_EC=3, d_clutter=2, clutter_density=.5)
    AfV = np.kron(np.eye(M), [[np.sqrt(1 - af), 0], [0, np.sqrt(af)]]); AfH = np.kron(np.eye(M), [[np.sqrt(af), 0], [0, np.sqrt(1 - af)]])
    ArV = np.kron(np.eye(M), [[np.sqrt(1 - ar), 0], [0, np.sqrt(ar)]]); ArH = np.kron(np.eye(M), [[np.sqrt(ar), 0], [0, np.sqrt(1 - ar)]])
    # Draw one UE geometry and determine its LoS/NLoS channel state.
    UE = functionUEconfiguration(K, cfg, wavelength, scatter, height)
    kappa = -cfg.d_clutter / np.log(1 - cfg.clutter_density) * (height - UE.positions[:, 2]) / (cfg.h_EC - UE.positions[:, 2])
    Kbar = 10 ** ((13 - .03 * UE.d3D) / 10); Kbar[np.random.rand(K) >= np.exp(-UE.d2D / kappa)] = 0
    beta = np.where(Kbar > 0, pl1(UE.d3D), pl2(UE.d3D))
    h = np.column_stack([np.sqrt(beta[k]) * functionSpatialSignature3DLoS(posDP, UE.phi[k], UE.theta[k], wavelength) for k in range(K)])
    hDP = np.kron(h, np.ones((2, 1))); hV, hH = AfV @ hDP, AfH @ hDP
    h1 = np.column_stack([np.sqrt(beta[k]) * functionSpatialSignature3DLoS(posSP, UE.phi_antenna1[k], UE.theta_antenna1[k], wavelength) for k in range(K)])
    h2 = np.column_stack([np.sqrt(beta[k]) * functionSpatialSignature3DLoS(posSP, UE.phi_antenna2[k], UE.theta_antenna2[k], wavelength) for k in range(K)])
    Rsp = np.empty((2 * M, 2 * M, K), complex); Rue = np.empty((2, 2, K), complex); RV = np.empty_like(Rsp); RH = np.empty_like(Rsp)
    for k in range(K):
        args = UE.phi[k], UE.dphi[k], UE.theta[k], UE.dtheta[k], "Uniform"
        Rbs = beta[k] * functionRlocalscattering3D(Mh, Mv, d, d, *args)
        Rue[:, :, k] = functionRlocalscattering3D(2, 1, d, d, *args); Rsp[:, :, k] = beta[k] * functionRlocalscattering3D(2 * Mh, Mv, d, d, *args)
        C = np.kron(Rbs, np.eye(2)); RV[:, :, k] = ArV @ C @ ArV.conj().T / (1 + Kbar[k]); RH[:, :, k] = ArH @ C @ ArH.conj().T / (1 + Kbar[k])
    # Estimate the DP and SP channels from orthogonal pilots.
    eV, eH, BV, BH = functionChannelEstimationDP(K, M, rho, RV, RH, Kbar, hV, hH, noise)
    e1, e2, B1, B2 = functionChannelEstimationSP(K, M, rho, Rsp, Rue, Kbar, h1, h2, noise)
    # Build only the receiver and power-control branches required by this figure.
    configs = ((eV, eH, BV, BH, "DP-noSIC", functionComputeSEDPnoSIC, functionPowerControlDPnoSIC, functionPowerControlDPnoSIC_Random),
               (e1, e2, B1, B2, "SP-SIC", functionComputeSESPSIC, functionPowerControlSPSIC, functionPowerControlSPSIC_Random),
               (e1, e2, B1, B2, "SP-noSIC", functionComputeSEDPnoSIC, functionPowerControlDPnoSIC, functionPowerControlDPnoSIC_Random))
    fixed, random, equal = [], [], []
    p = np.ones(K) * Pk / 2
    for a, b, A, B, scheme, compute, power, power_random in configs:
        q1, q2 = functionMMSEDetector(a, b, A, B, M, K, noise, scheme)
        AV = functionGenerateAV(q1, q2, a, b, A, B, K)
        equal.append(compute(AV, p, p, noise * 1e10, Nth, Eth).SE)
        out = power(AV, K, Pk, noise * 1e10, Eth, Nth); fixed.append((out[1], out[4]))
        out = power_random(AV, K, Pk, noise * 1e10, Eth, Nth); random.append((out[1], out[4]))
    for y in fixed: plt.plot(np.arange(1, 101), y[0])
    for value in equal: plt.plot(np.arange(1, 101), np.ones(100) * value)
    names = ["DP-noSIC", "SP-SCI", "SP-noSIC", "DP-noSIC (iter gaps)", "SP-SCI (itergaps)", "SP-noSIC (itergaps)"]
    # Export or return only the data series plotted in the corresponding figure.
    pd.DataFrame(np.column_stack(([x[0] for x in fixed] + [x[1] for x in fixed])), columns=names).to_excel("fixed_initial_value.xlsx", index=False)
    pd.DataFrame(np.column_stack(([x[0] for x in random] + [x[1] for x in random])), columns=names).to_excel("random_initial_value.xlsx", index=False)
    pd.DataFrame([equal], columns=["DPnoSIC", "SPSIC", "SPnoSIC"]).to_excel("equal_power_value.xlsx", index=False)
    plt.show()


if __name__ == "__main__": main()
