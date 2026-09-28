from __future__ import annotations

import numpy as np
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union


# ---------------------------------------------------------------------------
# Dataclasses — clean parameter containers
# ---------------------------------------------------------------------------


@dataclass
class GearParams:
    """Physical parameters of one gear stage."""

    teeth: int  # Z  — number of teeth
    speed_rpm: float  # n  — shaft speed [RPM]

    @property
    def shaft_freq(self) -> float:
        """f_r = n / 60  [Hz]"""
        return self.speed_rpm / 60.0

    @property
    def mesh_freq(self) -> float:
        """f_mesh = Z * f_r  [Hz]"""
        return self.teeth * self.shaft_freq

    @property
    def shaft_period(self) -> float:
        """T_r = 1 / f_r  [s]"""
        return 1.0 / self.shaft_freq


@dataclass
class EpicyclicGearParams(GearParams):
    """
    Physical parameters of an epicyclic (planetary) gear system.

    Topology assumed: fixed ring, carrier as input shaft, sun as output.

    Kinematics (Willis equation):
        f_sun      = f_c * (Z_r + Z_s) / Z_s
        f_planet   = f_c * Z_r / (2 * Z_p)   [planet spin in carrier frame]

    Two distinct mesh frequencies:
        f_mesh_ring = f_c * Z_r               [ring–planet contacts]
        f_mesh_sun  = f_c * (Z_r + Z_s)       [sun–planet contacts]
            = f_sun * Z_s  ✓
    """

    N_p: int = 3  # Number of planet gears
    Z_s: int = 20  # Sun teeth
    Z_p: int = 30  # Planet teeth
    Z_r: int = 80  # Ring teeth

    # ── derived shaft frequencies ────────────────────────────────────
    @property
    def carrier_freq(self) -> float:
        """f_c = carrier_rpm / 60  [Hz]  (speed_rpm IS the carrier speed)"""
        return self.speed_rpm / 60.0

    @property
    def sun_freq(self) -> float:
        """f_sun = f_c * (Z_r + Z_s) / Z_s  [Hz]  (Willis, fixed ring)"""
        return self.carrier_freq * (self.Z_r + self.Z_s) / self.Z_s

    @property
    def planet_spin_freq(self) -> float:
        """f_p = f_c * Z_r / (2 * Z_p)  [Hz]  (planet rotation in carrier frame)"""
        return self.carrier_freq * self.Z_r / (2 * self.Z_p)

    @property
    def planet_spin_period(self) -> float:
        """T_p = 1 / f_p  [s]"""
        return 1.0 / self.planet_spin_freq

    # ── two mesh frequencies ─────────────────────────────────────────
    @property
    def mesh_freq_ring(self) -> float:
        """f_mesh_ring = f_c * Z_r  [Hz]  (ring–planet mesh)"""
        return self.carrier_freq * self.Z_r

    @property
    def mesh_freq_sun(self) -> float:
        """f_mesh_sun = f_c * (Z_r + Z_s)  [Hz]  (sun–planet mesh)"""
        return self.carrier_freq * (self.Z_r + self.Z_s)

    # ── keep legacy property so old code that reads .epicyclic_mesh_freq
    #    still works (returns ring mesh freq) ─────────────────────────
    @property
    def epicyclic_mesh_freq(self) -> float:
        """Alias for mesh_freq_ring (backward compat)."""
        return self.mesh_freq_ring


@dataclass
class HarmonicParams:
    """Per-mesh-harmonic amplitude and phase."""

    M: int = 5  # number of mesh harmonics
    amplitudes: Optional[np.ndarray] = None
    phases: Optional[np.ndarray] = None

    def resolve(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        A = (
            self.amplitudes
            if self.amplitudes is not None
            else np.array([1.0 / m for m in range(1, self.M + 1)])
        )
        φ = (
            self.phases
            if self.phases is not None
            else rng.uniform(0, 2 * np.pi, size=self.M)
        )
        return A, φ


@dataclass
class ModulationParams:
    """Sideband modulation parameters."""

    K: int = 3  # sidebands per harmonic
    alpha_scale: float = 1  # AM depth (α_mk ~ U[0, alpha_scale])
    beta_scale: float = 0.2  # FM depth in radians (β_mk ~ U[0, beta_scale])
    sideband_decay: float = 0.5


@dataclass
class FaultParams:
    """Localized tooth fault (spall / crack) parameters."""

    enabled: bool = False
    impulse_amplitude: float = 1.0
    resonance_freq: float = 3000.0
    damping_ratio: float = 0.08
    amplitude_variation: float = 0.05


@dataclass
class SignalConfig:
    """Top-level signal generation configuration."""

    duration: float = 2.0
    sampling_rate: int = 25_600
    gear: Union[GearParams, EpicyclicGearParams] = field(
        default_factory=lambda: GearParams(teeth=20, speed_rpm=1500)
    )
    harmonic: HarmonicParams = field(default_factory=HarmonicParams)
    modulation: ModulationParams = field(default_factory=ModulationParams)
    fault: FaultParams = field(default_factory=FaultParams)
    noise_std: float = 0.02
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class GearSignalGenerator:
    """
    Generates synthetic gear vibration signals implementing the McFadden–Smith model.
    Supports both Spur and Epicyclic gear configurations.

    For epicyclic gears, two physically distinct mesh frequencies are synthesized:
        • Ring–planet mesh  :  f_mesh_ring = f_c * Z_r
        • Sun–planet mesh   :  f_mesh_sun  = f_c * (Z_r + Z_s)
    Each is independently modulated and windowed by the planet passage function w_i(t).
    """

    def __init__(self, config: SignalConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self._validate()

    def generate(self) -> dict:
        cfg = self.cfg
        N = int(cfg.sampling_rate * cfg.duration)
        t = np.linspace(0, cfg.duration, N, endpoint=False)

        if isinstance(cfg.gear, EpicyclicGearParams):
            x_mesh = self._generate_epicyclic_mesh(t)
            T_fault = cfg.gear.planet_spin_period
        else:
            x_mesh = self._mesh_harmonics(t)
            T_fault = cfg.gear.shaft_period

        x_fault = self._fault_impulse(t, T_fault) if cfg.fault.enabled else np.zeros(N)
        x_noise = (
            self.rng.normal(0.0, cfg.noise_std, size=N)
            if cfg.noise_std > 0
            else np.zeros(N)
        )

        signal = x_mesh + x_fault + x_noise

        return {
            "t": t,
            "signal": signal,
            "x_mesh": x_mesh,
            "x_fault": x_fault,
            "x_noise": x_noise,
            "metadata": self._metadata(),
        }

    # ------------------------------------------------------------------
    # Spur gear mesh synthesis
    # ------------------------------------------------------------------

    def _mesh_harmonics(
        self,
        t: np.ndarray,
        f_mesh: Optional[float] = None,
        f_r: Optional[float] = None,
    ) -> np.ndarray:
        """
        McFadden-Smith synthesis for ONE mesh frequency:
            Σ_m  A_m [1 + a_m(t)] cos(2π m f_mesh t + φ_m + b_m(t))
        """
        cfg = self.cfg
        f_mesh_val = f_mesh if f_mesh is not None else cfg.gear.mesh_freq
        f_r_val = f_r if f_r is not None else cfg.gear.shaft_freq
        nyq = cfg.sampling_rate / 2.0
        A, phi = cfg.harmonic.resolve(self.rng)

        signal = np.zeros_like(t)
        for idx, m in enumerate(range(1, cfg.harmonic.M + 1)):
            f_h = m * f_mesh_val
            if f_h >= nyq:
                break
            a_m = self._am_modulation(t, m, f_r_val)
            b_m = self._fm_modulation(t, m, f_r_val)
            signal += (
                A[idx] * (1.0 + a_m) * np.cos(2 * np.pi * f_h * t + phi[idx] + b_m)
            )
        return signal

    # ------------------------------------------------------------------
    # Epicyclic mesh synthesis  (fixed ring, carrier input)
    # ------------------------------------------------------------------

    def _generate_epicyclic_mesh(self, t: np.ndarray) -> np.ndarray:
        """
        Spatio-temporal synthesis with TWO mesh components per planet:

            x(t) = Σ_i  w_i(t) * [ f_ring_i(t) + f_sun_i(t) ]

        where
            w_i(t)      — planet passage window (raised cosine)
            f_ring_i(t) — ring–planet mesh harmonics at f_mesh_ring = f_c * Z_r
            f_sun_i(t)  — sun–planet mesh harmonics  at f_mesh_sun  = f_c * (Z_r + Z_s)

        Modulation sidebands use:
            • f_c  (carrier) for the ring–planet component
            • f_sun         for the sun–planet component
        so that Z_s, Z_p, and Z_r all influence the synthesized signal.
        """
        gear = self.cfg.gear  # type: EpicyclicGearParams
        N = len(t)
        f_c = gear.carrier_freq
        f_sun = gear.sun_freq  # f_c * (Z_r + Z_s) / Z_s

        f_mesh_ring = gear.mesh_freq_ring  # f_c * Z_r
        f_mesh_sun = gear.mesh_freq_sun  # f_c * (Z_r + Z_s)

        total = np.zeros(N)

        for i in range(1, gear.N_p + 1):
            # ── planet passage window w_i(t) ──────────────────────────
            psi_i = 2 * np.pi * (i - 1) / gear.N_p
            angle = 2 * np.pi * f_c * t - psi_i
            angle_norm = (angle + np.pi) % (2 * np.pi) - np.pi

            w_i = np.zeros(N)
            mask = np.abs(angle_norm) <= np.pi / gear.N_p
            w_i[mask] = 0.5 * (1.0 + np.cos(angle_norm[mask]))

            # ── ring–planet mesh  (modulated by carrier f_c) ──────────
            f_ring_i = self._mesh_harmonics(t, f_mesh=f_mesh_ring, f_r=f_c)

            # ── sun–planet mesh   (modulated by sun freq f_sun) ────────
            f_sun_i = self._mesh_harmonics(t, f_mesh=f_mesh_sun, f_r=f_sun)

            total += w_i * (f_ring_i + f_sun_i)

        return total

    # ------------------------------------------------------------------
    # Modulation helpers
    # ------------------------------------------------------------------

    def _am_modulation(self, t: np.ndarray, m: int, f_r: float) -> np.ndarray:
        mp = self.cfg.modulation
        a = np.zeros_like(t)
        for k in range(1, mp.K + 1):
            alpha = self.rng.uniform(mp.alpha_scale / 2, mp.alpha_scale) * (
                mp.sideband_decay ** (k - 1)
            )
            psi = self.rng.uniform(0, 2 * np.pi)
            a += alpha * np.cos(2 * np.pi * k * f_r * t + psi)
        return a

    def _fm_modulation(self, t: np.ndarray, m: int, f_r: float) -> np.ndarray:
        mp = self.cfg.modulation
        b = np.zeros_like(t)
        for k in range(1, mp.K + 1):
            beta = self.rng.uniform(0, mp.beta_scale) * (mp.sideband_decay ** (k - 1))
            gamma = self.rng.uniform(0, 2 * np.pi)
            b += beta * np.cos(2 * np.pi * k * f_r * t + gamma)
        return b

    # ------------------------------------------------------------------
    # Fault impulse train
    # ------------------------------------------------------------------

    def _fault_impulse(self, t: np.ndarray, T_fault: float) -> np.ndarray:
        fp = self.cfg.fault
        fs = self.cfg.sampling_rate
        N = len(t)

        omega_n = 2 * np.pi * fp.resonance_freq
        zeta = fp.damping_ratio
        omega_d = omega_n * np.sqrt(max(1 - zeta**2, 1e-6))

        kernel_len = min(int(T_fault * fs * 3), N // 2)
        t_kernel = np.arange(kernel_len) / fs
        h = np.exp(-zeta * omega_n * t_kernel) * np.sin(omega_d * t_kernel)
        h /= np.max(np.abs(h)) + 1e-12

        fault_comb = np.zeros(N)
        impact_times = np.arange(0, self.cfg.duration, T_fault)
        for t_i in impact_times:
            idx = int(round(t_i * fs))
            if idx >= N:
                break
            C_i = fp.impulse_amplitude * (
                1.0 + self.rng.normal(0, fp.amplitude_variation)
            )
            fault_comb[idx] += C_i

        return np.convolve(fault_comb, h, mode="full")[:N]

    # ------------------------------------------------------------------
    # Validation & metadata
    # ------------------------------------------------------------------

    def _validate(self):
        cfg = self.cfg
        nyq = cfg.sampling_rate / 2.0

        if isinstance(cfg.gear, EpicyclicGearParams):
            g = cfg.gear
            # Check both mesh frequencies
            for label, f_m in [
                ("ring–planet", g.mesh_freq_ring),
                ("sun–planet", g.mesh_freq_sun),
            ]:
                f_max = cfg.harmonic.M * f_m
                if f_max >= nyq:
                    warnings.warn(
                        f"Epicyclic {label} mesh: highest harmonic "
                        f"{f_max:.1f} Hz ≥ Nyquist {nyq:.1f} Hz. "
                        f"Increase sampling_rate or reduce M / carrier_rpm."
                    )
        else:
            f_max = cfg.harmonic.M * cfg.gear.mesh_freq
            if f_max >= nyq:
                warnings.warn(
                    f"Highest harmonic {f_max:.1f} Hz ≥ Nyquist {nyq:.1f} Hz. "
                    "Increase sampling_rate or reduce M."
                )

    def _metadata(self) -> dict:
        g = self.cfg.gear
        meta = {
            "M": self.cfg.harmonic.M,
            "K": self.cfg.modulation.K,
            "faulty": self.cfg.fault.enabled,
            "resonance_hz": self.cfg.fault.resonance_freq,
            "damping_ratio": self.cfg.fault.damping_ratio,
        }
        if isinstance(g, EpicyclicGearParams):
            meta.update(
                {
                    "carrier_freq_hz": g.carrier_freq,
                    "sun_freq_hz": g.sun_freq,
                    "planet_spin_freq_hz": g.planet_spin_freq,
                    "mesh_freq_ring_hz": g.mesh_freq_ring,
                    "mesh_freq_sun_hz": g.mesh_freq_sun,
                    "Z_s": g.Z_s,
                    "Z_p": g.Z_p,
                    "Z_r": g.Z_r,
                    "N_p": g.N_p,
                }
            )
        else:
            meta.update(
                {
                    "shaft_freq_hz": g.shaft_freq,
                    "mesh_freq_hz": g.mesh_freq,
                }
            )
        return meta


# ---------------------------------------------------------------------------
# Convenience factories — backward compatibility
# ---------------------------------------------------------------------------


def generate_spur_gear_signal(
    duration: float = 1.0,
    sampling_rate: int = 25_600,
    gear_speed_rpm: float = 1500,
    gear_teeth: int = 20,
    faulty: bool = False,
    M: int = 5,
    K: int = 3,
    noise_level: float = 0.02,
    impulse_amplitude: float = 1.0,
    resonance_freq: float = 3000.0,
    damping_ratio: float = 0.08,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = SignalConfig(
        duration=duration,
        sampling_rate=sampling_rate,
        gear=GearParams(teeth=gear_teeth, speed_rpm=gear_speed_rpm),
        harmonic=HarmonicParams(M=M),
        modulation=ModulationParams(K=K),
        fault=FaultParams(
            enabled=faulty,
            impulse_amplitude=impulse_amplitude,
            resonance_freq=resonance_freq,
            damping_ratio=damping_ratio,
        ),
        noise_std=noise_level,
        seed=seed,
    )
    result = GearSignalGenerator(cfg).generate()
    return result["t"], result["signal"]


def generate_epicyclic_gear_signal(
    duration: float = 1.0,
    sampling_rate: int = 25_600,
    carrier_speed_rpm: float = 1000,
    N_p: int = 3,
    Z_s: int = 20,
    Z_p: int = 30,
    Z_r: int = 80,
    faulty: bool = False,
    M: int = 5,
    K: int = 3,
    noise_level: float = 0.02,
    impulse_amplitude: float = 1.0,
    resonance_freq: float = 3000.0,
    damping_ratio: float = 0.08,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = SignalConfig(
        duration=duration,
        sampling_rate=sampling_rate,
        gear=EpicyclicGearParams(
            speed_rpm=carrier_speed_rpm,
            teeth=Z_s,
            N_p=N_p,
            Z_s=Z_s,
            Z_p=Z_p,
            Z_r=Z_r,
        ),
        harmonic=HarmonicParams(M=M),
        modulation=ModulationParams(K=K),
        fault=FaultParams(
            enabled=faulty,
            impulse_amplitude=impulse_amplitude,
            resonance_freq=resonance_freq,
            damping_ratio=damping_ratio,
        ),
        noise_std=noise_level,
        seed=seed,
    )
    result = GearSignalGenerator(cfg).generate()
    return result["t"], result["signal"]
