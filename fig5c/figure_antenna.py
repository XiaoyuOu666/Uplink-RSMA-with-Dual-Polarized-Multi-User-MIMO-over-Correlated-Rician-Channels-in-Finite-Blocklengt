"""Generate the sum-SE curves versus the number of antenna pairs used in Fig. 5(c)."""

from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.io import savemat
from functionUEconfiguration import functionUEconfiguration
from functionSpatialSignature3DLoS import functionSpatialSignature3DLoS
from functionRlocalscattering3D import functionRlocalscattering3D
from functionChannelEstimationDP import functionChannelEstimationDP
from functionChannelEstimationSP import functionChannelEstimationSP
from functionMMSEDetector import functionMMSEDetector
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
    af = 1 / (10 ** (np.array([15, 15, 15]) / 10) + 1); ar = 1 / (10 ** (np.array([0, 10, 20]) / 10) + 1)
    # Define the DP and hardware-matched SP array geometries.
    Mv, Mhs, d, height, scatter = 8, np.arange(4, 21, 2), .5, 20, 20
    cfg = SimpleNamespace(InF_width=300, InF_length=300, h_EC=3, d_clutter=2, clutter_density=.5)
    dp = np.zeros((len(Mhs), 3, 500)); sp = np.zeros((len(Mhs), 500)); sic = np.zeros((len(Mhs), 500))
    # Average the requested figure quantities over independent channel realizations.
    for nor in range(500):
        # Draw one UE geometry and determine its LoS/NLoS channel state.
        UE = functionUEconfiguration(K, cfg, wavelength, scatter, height)
        kappa = -cfg.d_clutter / np.log(1 - cfg.clutter_density) * (height - UE.positions[:, 2]) / (cfg.h_EC - UE.positions[:, 2])
        Kbar = 10 ** ((13 - .03 * UE.d3D) / 10); Kbar[np.random.rand(K) >= np.exp(-UE.d2D / kappa)] = 0
        shadowing = 4 * np.random.randn(K)
        beta = np.where(Kbar > 0, pl1(UE.d3D), pl2(UE.d3D)) * 10 ** (-shadowing / 10)
        for mi, Mh in enumerate(Mhs):
            M = Mv * Mh; m = np.arange(M); m2 = np.arange(2 * M)
            posDP = np.vstack((np.zeros(M), m % Mh * d * wavelength, m // Mh * d * wavelength))
            posSP = np.vstack((np.zeros(2 * M), m2 % (2 * Mh) * d * wavelength, m2 // (2 * Mh) * d * wavelength))
            h = np.column_stack([np.sqrt(beta[k]) * functionSpatialSignature3DLoS(posDP, UE.phi[k], UE.theta[k], wavelength) for k in range(K)])
            hDP = np.kron(h, np.ones((2, 1)))
            h1 = np.column_stack([np.sqrt(beta[k]) * functionSpatialSignature3DLoS(posSP, UE.phi_antenna1[k], UE.theta_antenna1[k], wavelength) for k in range(K)])
            h2 = np.column_stack([np.sqrt(beta[k]) * functionSpatialSignature3DLoS(posSP, UE.phi_antenna2[k], UE.theta_antenna2[k], wavelength) for k in range(K)])
            hV = np.empty((2 * M, K, 3), complex); hH = np.empty_like(hV); RV = np.empty((2 * M, 2 * M, K, 3), complex); RH = np.empty_like(RV)
            Rsp = np.empty((2 * M, 2 * M, K), complex); Rue = np.empty((2, 2, K), complex)
            AfV = [np.kron(np.eye(M), [[np.sqrt(1 - x), 0], [0, np.sqrt(x)]]) for x in af]
            AfH = [np.kron(np.eye(M), [[np.sqrt(x), 0], [0, np.sqrt(1 - x)]]) for x in af]
            ArV = [np.kron(np.eye(M), [[np.sqrt(1 - x), 0], [0, np.sqrt(x)]]) for x in ar]
            ArH = [np.kron(np.eye(M), [[np.sqrt(x), 0], [0, np.sqrt(1 - x)]]) for x in ar]
            for j in range(3): hV[:, :, j], hH[:, :, j] = AfV[j] @ hDP, AfH[j] @ hDP
            for k in range(K):
                args = UE.phi[k], UE.dphi[k], UE.theta[k], UE.dtheta[k], "Uniform"
                Rbs = beta[k] * functionRlocalscattering3D(Mh, Mv, d, d, *args)
                Rue[:, :, k] = functionRlocalscattering3D(2, 1, d, d, *args); Rsp[:, :, k] = beta[k] * functionRlocalscattering3D(2 * Mh, Mv, d, d, *args)
                C = np.kron(Rbs, np.eye(2))
                for j in range(3): RV[:, :, k, j] = ArV[j] @ C @ ArV[j].conj().T / (1 + Kbar[k]); RH[:, :, k, j] = ArH[j] @ C @ ArH[j].conj().T / (1 + Kbar[k])
            # Estimate the DP and SP channels from orthogonal pilots.
            eV, eH, BV, BH = functionChannelEstimationDP(K, M, rho, RV, RH, Kbar, hV, hH, noise)
            e1, e2, B1, B2 = functionChannelEstimationSP(K, M, rho, Rsp, Rue, Kbar, h1, h2, noise)
            for j in range(3):
                q1, q2 = functionMMSEDetector(eV[:, :, j], eH[:, :, j], BV[:, :, :, j], BH[:, :, :, j], M, K, noise, "DP-noSIC")
                AV = functionGenerateAV(q1, q2, eV[:, :, j], eH[:, :, j], BV[:, :, :, j], BH[:, :, :, j], K)
                dp[mi, j, nor] = functionPowerControlDPnoSIC(AV, K, Pk, noise * 1e10, Eth, Nth)[0]
            for out, scheme, power in ((sic, "SP-SIC", functionPowerControlSPSIC), (sp, "SP-noSIC", functionPowerControlDPnoSIC)):
                q1, q2 = functionMMSEDetector(e1, e2, B1, B2, M, K, noise, scheme)
                out[mi, nor] = power(functionGenerateAV(q1, q2, e1, e2, B1, B2, K), K, Pk, noise * 1e10, Eth, Nth)[0]
        print(f"this is {nor + 1} th iteration")
    # Export or return only the data series plotted in the corresponding figure.
    savemat("my_workspace.mat", {"max_SE_DPnoSIC_MMSE_nor": dp, "max_SE_SPSIC_MMSE_nor": sic, "max_SE_SPnoSIC_MMSE_nor": sp})
    a, b, c = dp.mean(2), sp.mean(1), sic.mean(1)
    pd.DataFrame(np.column_stack((a, b, c)), columns=["DP-noSIC_XPDr_0", "DP-noSIC_XPDr_10", "DP-noSIC_XPDr_20", "SP-noSIC", "SP-localSIC"]).to_excel("Antenna2.xlsx", index=False)


if __name__ == "__main__": main()
