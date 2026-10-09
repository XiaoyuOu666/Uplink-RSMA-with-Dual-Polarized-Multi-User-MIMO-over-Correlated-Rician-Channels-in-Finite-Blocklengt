"""Generate the sum-SE CDF data for the detector comparison in Fig. 4(b)."""

from types import SimpleNamespace
import numpy as np
import pandas as pd
from functionUEconfiguration import functionUEconfiguration
from functionSpatialSignature3DLoS import functionSpatialSignature3DLoS
from functionRlocalscattering3D import functionRlocalscattering3D
from functionChannelEstimationDP import functionChannelEstimationDP
from functionChannelEstimationSP import functionChannelEstimationSP
from functionMMSEDetector import functionMMSEDetector
from functionMRDetector import functionMRDetector
from functionRZFDetector import functionRZFDetector
from functionGenerateAV import functionGenerateAV
from functionPowerControlDPnoSIC import functionPowerControlDPnoSIC
from functionPowerControlSPSIC import functionPowerControlSPSIC


def main():
    """Run the simulation and return or export only the data required by the corresponding figure."""
    # Set the common system, reliability, and Monte Carlo parameters.
    K, fc, Nth, Eth = 8, 3.5, 500, 1e-5
    wavelength = 299792458 / (fc * 1e9); noise = 10 ** ((-84 - 30) / 10); rho = Pk = 10 ** ((20 - 30) / 10)
    pl1 = lambda d: 10 ** -(3.184 + 2.15 * np.log10(d) + 1.9 * np.log10(fc))
    pl2 = lambda d: 10 ** -(3.363 + 2.19 * np.log10(d) + 2 * np.log10(fc))
    af, ar = 1 / (10 ** 1.5 + 1), 1 / (10 ** .5 + 1)
    # Define the DP and hardware-matched SP array geometries.
    Mv, Mh, M, d, height, scatter = 8, 4, 32, .5, 20, 20
    m = np.arange(M); m2 = np.arange(2 * M)
    posDP = np.vstack((np.zeros(M), m % Mh * d * wavelength, m // Mh * d * wavelength))
    posSP = np.vstack((np.zeros(2 * M), m2 % Mv * d * wavelength, m2 // (2 * Mh) * d * wavelength))
    cfg = SimpleNamespace(InF_width=300, InF_length=300, h_EC=3, d_clutter=2, clutter_density=.5)
    AfV = np.kron(np.eye(M), [[np.sqrt(1 - af), 0], [0, np.sqrt(af)]]); AfH = np.kron(np.eye(M), [[np.sqrt(af), 0], [0, np.sqrt(1 - af)]])
    ArV = np.kron(np.eye(M), [[np.sqrt(1 - ar), 0], [0, np.sqrt(ar)]]); ArH = np.kron(np.eye(M), [[np.sqrt(ar), 0], [0, np.sqrt(1 - ar)]])
    samples = np.empty((500, 3, 3))
    # Average the requested figure quantities over independent channel realizations.
    for nor in range(500):
        # Draw one UE geometry and determine its LoS/NLoS channel state.
        UE = functionUEconfiguration(K, cfg, wavelength, scatter, height)
        shadowing = 4 * np.random.randn(K)
        kappa = -cfg.d_clutter / np.log(1 - cfg.clutter_density) * (height - UE.positions[:, 2]) / (cfg.h_EC - UE.positions[:, 2])
        Kbar = 10 ** ((13 - .03 * UE.d3D) / 10); Kbar[np.random.rand(K) >= np.exp(-UE.d2D / kappa)] = 0
        beta = np.where(Kbar > 0, pl1(UE.d3D), pl2(UE.d3D)) * 10 ** (-shadowing / 10)
        h = np.column_stack([np.sqrt(beta[k]) * functionSpatialSignature3DLoS(posDP, UE.phi[k], UE.theta[k], wavelength) for k in range(K)])
        hDP = np.kron(h, np.ones((2, 1))); hV, hH = AfV @ hDP, AfH @ hDP
        h1 = np.column_stack([np.sqrt(beta[k]) * functionSpatialSignature3DLoS(posSP, UE.phi_antenna1[k], UE.theta_antenna1[k], wavelength) for k in range(K)])
        h2 = np.column_stack([np.sqrt(beta[k]) * functionSpatialSignature3DLoS(posSP, UE.phi_antenna2[k], UE.theta_antenna2[k], wavelength) for k in range(K)])
        R = np.empty((4 * M, 4 * M, K), complex); Rsp = np.empty((2 * M, 2 * M, K), complex); Rue = np.empty((2, 2, K), complex)
        for k in range(K):
            args = UE.phi[k], UE.dphi[k], UE.theta[k], UE.dtheta[k], "Uniform"
            Rbs = beta[k] * functionRlocalscattering3D(Mh, Mv, d, d, *args)
            Rue[:, :, k] = functionRlocalscattering3D(2, 1, d, d, *args); Rsp[:, :, k] = beta[k] * functionRlocalscattering3D(2 * Mh, Mv, d, d, *args)
            C = np.kron(Rbs, np.eye(2)); RV = ArV @ C @ ArV.conj().T / (1 + Kbar[k]); RH = ArH @ C @ ArH.conj().T / (1 + Kbar[k])
            R[:, :, k] = np.block([[RV, np.zeros_like(RV)], [np.zeros_like(RH), RH]])
        # Estimate the DP and SP channels from orthogonal pilots.
        eV, eH, BV, BH = functionChannelEstimationDP(K, M, rho, R, Kbar, hV, hH, noise)
        e1, e2, B1, B2 = functionChannelEstimationSP(K, M, rho, Rsp, Rue, Kbar, h1, h2, noise)
        # Build only the receiver and power-control branches required by this figure.
        schemes = ((eV, eH, BV, BH, "DP-noSIC", functionPowerControlDPnoSIC),
                   (e1, e2, B1, B2, "SP-noSIC", functionPowerControlDPnoSIC),
                   (e1, e2, B1, B2, "SP-SIC", functionPowerControlSPSIC))
        for si, (a, b, A, B, scheme, power) in enumerate(schemes):
            for di, detector in enumerate((functionMMSEDetector, functionMRDetector, functionRZFDetector)):
                if di == 0: q1, q2 = detector(a, b, A, B, M, K, noise, scheme)
                elif di == 1: q1, q2 = detector(a, b, K)
                else: q1, q2 = detector(a, b, noise, K, M)
                AV = functionGenerateAV(q1, q2, a, b, A, B, K)
                samples[nor, si, di] = power(AV, K, Pk, noise * 1e10, Eth, Nth)[0]
        print(f"this is {nor + 1} th iteration")
    low = np.floor(samples.min() / 5) * 5 - 1; high = np.floor(samples.max() / 5) * 5 + 5; x = np.linspace(low + (high - low) / 500, high, 500)
    order = [(0, 0), (0, 2), (0, 1), (1, 0), (1, 2), (1, 1), (2, 0), (2, 2), (2, 1)]
    data = [np.mean(samples[:, s, d, None] < x, axis=0) for s, d in order] + [x]
    names = ["DPnoSIC_MMSE_CDF", "DPnoSIC_RZF_CDF", "DPnoSIC_MR_CDF", "SPnoSIC_MMSE_CDF", "SPnoSIC_RZF_CDF", "SPnoSIC_MR_CDF", "SPSIC_MMSE_CDF", "SPSIC_RZF_CDF", "SPSIC_MR_CDF", "x-axis"]
    # Export or return only the data series plotted in the corresponding figure.
    pd.DataFrame(np.column_stack(data), columns=names).to_excel("CDF_detector_modify.xlsx", index=False)


if __name__ == "__main__": main()
