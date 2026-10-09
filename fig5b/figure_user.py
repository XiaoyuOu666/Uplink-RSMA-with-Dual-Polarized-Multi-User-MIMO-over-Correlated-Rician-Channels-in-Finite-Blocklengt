"""Generate the sum-SE curves versus the number of users used in Fig. 5(b)."""

from types import SimpleNamespace
import numpy as np
import pandas as pd
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
    K_multi = np.arange(2, 17, 2)
    fc, Nth, Eth = 3.5, 500, 1e-5
    wavelength = 299792458 / (fc * 1e9)
    Noise_power, rho, Pk = 10 ** ((-84 - 30) / 10), 10 ** ((20 - 30) / 10), 10 ** ((20 - 30) / 10)
    PL_LoS = lambda d: 10 ** -(3.184 + 2.15 * np.log10(d) + 1.9 * np.log10(fc))
    PL_NLoS = lambda d: 10 ** -(3.363 + 2.19 * np.log10(d) + 2 * np.log10(fc))
    alpha_f, alpha_r = 1 / (10 ** 1.5 + 1), 1 / (10 ** .5 + 1)
    # Define the DP and hardware-matched SP array geometries.
    Mv, Mh, d_H, d_V, BS_height, r_scatter = 8, 4, .5, .5, 20, 20
    M = Mv * Mh
    m = np.arange(M); m2 = np.arange(2 * M)
    pos_DP = np.vstack((np.zeros(M), m % Mh * d_H * wavelength, m // Mh * d_H * wavelength))
    pos_SP = np.vstack((np.zeros(2 * M), m2 % Mv * d_H * wavelength, m2 // (2 * Mh) * d_H * wavelength))
    cfg = SimpleNamespace(InF_width=300, InF_length=300, h_EC=3, d_clutter=2, clutter_density=.5)
    realizations = 500
    values = np.zeros((len(K_multi), 3, 3))
    AfV = np.kron(np.eye(M), [[np.sqrt(1 - alpha_f), 0], [0, np.sqrt(alpha_f)]])
    AfH = np.kron(np.eye(M), [[np.sqrt(alpha_f), 0], [0, np.sqrt(1 - alpha_f)]])
    ArV = np.kron(np.eye(M), [[np.sqrt(1 - alpha_r), 0], [0, np.sqrt(alpha_r)]])
    ArH = np.kron(np.eye(M), [[np.sqrt(alpha_r), 0], [0, np.sqrt(1 - alpha_r)]])

    # Average the requested figure quantities over independent channel realizations.
    for nor in range(realizations):
        for nou, K in enumerate(K_multi):
            # Draw one UE geometry and determine its LoS/NLoS channel state.
            UE = functionUEconfiguration(K, cfg, wavelength, r_scatter, BS_height)
            shadowing = 4 * np.random.randn(K)
            kappa = -cfg.d_clutter / np.log(1 - cfg.clutter_density) * (BS_height - UE.positions[:, 2]) / (cfg.h_EC - UE.positions[:, 2])
            mixed = np.random.rand(K) < np.exp(-UE.d2D / kappa)
            Rbs0 = np.empty((M, M, K), complex); Rue = np.empty((2, 2, K), complex); Rsp0 = np.empty((2 * M, 2 * M, K), complex)
            for k in range(K):
                args = UE.phi[k], UE.dphi[k], UE.theta[k], UE.dtheta[k], "Uniform"
                Rbs0[:, :, k] = functionRlocalscattering3D(Mh, Mv, d_H, d_V, *args); Rue[:, :, k] = functionRlocalscattering3D(2, 1, d_H, d_V, *args); Rsp0[:, :, k] = functionRlocalscattering3D(2 * Mh, Mv, d_H, d_V, *args)
            for fading, los in enumerate((mixed, np.zeros(K, bool), np.ones(K, bool))):
                Kbar = 10 ** ((13 - .03 * UE.d3D) / 10); Kbar[~los] = 0
                beta = np.where(los, PL_LoS(UE.d3D), PL_NLoS(UE.d3D)) * 10 ** (-shadowing / 10)
                h = np.column_stack([np.sqrt(beta[k]) * functionSpatialSignature3DLoS(pos_DP, UE.phi[k], UE.theta[k], wavelength) for k in range(K)])
                hDP = np.kron(h, np.ones((2, 1)))
                h1 = np.column_stack([np.sqrt(beta[k]) * functionSpatialSignature3DLoS(pos_SP, UE.phi_antenna1[k], UE.theta_antenna1[k], wavelength) for k in range(K)])
                h2 = np.column_stack([np.sqrt(beta[k]) * functionSpatialSignature3DLoS(pos_SP, UE.phi_antenna2[k], UE.theta_antenna2[k], wavelength) for k in range(K)])
                hV, hH = AfV @ hDP, AfH @ hDP
                Rbs = np.empty((M, M, K), complex); Rsp = np.empty((2 * M, 2 * M, K), complex)
                RV = np.empty_like(Rsp); RH = np.empty_like(Rsp)
                for k in range(K):
                    Rbs[:, :, k] = beta[k] * Rbs0[:, :, k]
                    Rsp[:, :, k] = beta[k] * Rsp0[:, :, k]
                    C = np.kron(Rbs[:, :, k], np.eye(2))
                    RV[:, :, k] = ArV @ C @ ArV.conj().T / (1 + Kbar[k])
                    RH[:, :, k] = ArH @ C @ ArH.conj().T / (1 + Kbar[k])
                # Estimate the DP and SP channels from orthogonal pilots.
                eV, eH, BV, BH = functionChannelEstimationDP(K, M, rho, RV, RH, Kbar, hV, hH, Noise_power)
                e1, e2, B1, B2 = functionChannelEstimationSP(K, M, rho, Rsp, Rue, Kbar, h1, h2, Noise_power)
                results = []
                for channels, errors, scheme, power in (([eV, eH], [BV, BH], "DP-noSIC", functionPowerControlDPnoSIC),
                                                         ([e1, e2], [B1, B2], "SP-noSIC", functionPowerControlDPnoSIC),
                                                         ([e1, e2], [B1, B2], "SP-SIC", functionPowerControlSPSIC)):
                    q1, q2 = functionMMSEDetector(*channels, *errors, M, K, Noise_power, scheme)
                    AV = functionGenerateAV(q1, q2, *channels, *errors, K)
                    results.append(power(AV, K, Pk, Noise_power * 1e10, Eth, Nth)[0])
                values[nou, fading] += results
        print(f"this is {nor + 1} th iteration")
    columns = [f"{scheme}_{fading}" for scheme in ("DPnoSIC", "SPnoSIC", "SPSIC") for fading in ("Mixed", "Rayleigh", "Rician")]
    # Export or return only the data series plotted in the corresponding figure.
    pd.DataFrame(np.hstack((values[:, :, 0], values[:, :, 1], values[:, :, 2])) / realizations, columns=columns).to_excel("user_modify.xlsx", index=False)


if __name__ == "__main__":
    main()
