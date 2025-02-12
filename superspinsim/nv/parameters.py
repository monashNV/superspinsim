import math


nv_parameters = {}

bohr_magneton_gyro = math.tau*13.9962449171e6
r"""
The Bohr magneton divided by the reduced Plank constant.

Units
-----
Radians per second per Tesla.

Reference
---------
Table XXXIII, 2022 CODATA recommended values
"""

nv_parameters["bohr_magneton_gyro"] = {
    "value": bohr_magneton_gyro,
    "units": "Radians per second per Tesla",
    "references": [
        "Table XXXIII, 2022 CODATA recommended values"
    ]
}

# Ground state ----------------------------------------------------------------

zero_field_splitting_ground = math.tau*2.872e9
r"""
D_g

Units
-----
Radians per second.

Reference
---------
Section IV, Phys. Rev. B 79 075203
"""

nv_parameters["zero_field_splitting_ground"] = {
    "value": zero_field_splitting_ground,
    "units": "Radians per second",
    "references": [
        "Section IV, Phys. Rev. B 79 075203"
    ]
}

longitudinal_g_factor_ground = 2.0029
r"""
g_g^\parallel

References
----------
Section IV, Phys. Rev. B 79 075203
Table 2, Physics Reports 528 1–45
"""

nv_parameters["longitudinal_g_factor_ground"] = {
    "value": longitudinal_g_factor_ground,
    "units": "1",
    "references": [
        "Section IV, Phys. Rev. B 79 075203",
        "Table 2, Physics Reports 528 1–45"
    ]
}

longitudinal_gyromagnetic_ratio_ground = longitudinal_g_factor_ground\
    * bohr_magneton_gyro
r"""
\gamma_g^\parallel

Units
-----
Radians per second per Tesla
"""

nv_parameters["longitudinal_gyromagnetic_ratio_ground"] = {
    "value": longitudinal_gyromagnetic_ratio_ground,
    "units": "Radians per second per Tesla",
}

transverse_g_factor_ground = 2.0031
r"""
g_g^\perp

References
----------
Section IV, Phys. Rev. B 79 075203
Table 2, Physics Reports 528 1–45
"""

nv_parameters["transverse_g_factor_ground"] = {
    "value": transverse_g_factor_ground,
    "units": "1",
    "references": [
        "Section IV, Phys. Rev. B 79 07520,",
        "Table 2, Physics Reports 528 1–45"
    ]
}

transverse_gyromagnetic_ratio_ground = transverse_g_factor_ground\
    * bohr_magneton_gyro
r"""
\gamma_g^\perp

Units
-----
Radians per second per Tesla
"""

nv_parameters["transverse_gyromagnetic_ratio_ground"] = {
    "value": transverse_gyromagnetic_ratio_ground,
    "units": "Radians per second per Tesla"
}

spin_lattice_relaxation_time_ground = 7.7e-3
r"""
T_{1, g}

Units
-----
Seconds

References
----------
Page 3, Phys. Rev. Lett. 101 047601
Table 11, Physics Reports 528 1–45
"""

nv_parameters["spin_lattice_relaxation_time_ground"] = {
    "value": spin_lattice_relaxation_time_ground,
    "units": "Seconds",
    "references": [
        "Page 3, Phys. Rev. Lett. 101 047601",
        "Table 11, Physics Reports 528 1–45"
    ]
}

spin_spin_relaxation_time_ground = 6.7e-6
r"""
T_{2, g}

Units
-----
Seconds

References
----------
Page 3, Phys. Rev. Lett. 101 047601
Table 11, Physics Reports 528 1–45
"""

nv_parameters["spin_spin_relaxation_time_ground"] = {
    "value": spin_spin_relaxation_time_ground,
    "units": "Seconds",
    "references": [
        "Page 3, Phys. Rev. Lett. 101 047601",
        "Table 11, Physics Reports 528 1–45"
    ]
}

# Excited state ---------------------------------------------------------------

zero_field_splitting_excited = math.tau*1.42e9
r"""
D_e

Units
-----
Radians per second

References
----------
Figure 2, New J. Phys. 11 013017
Table 4, Physics Reports 528 1–45
"""

nv_parameters["zero_field_splitting_excited"] = {
    "value": zero_field_splitting_excited,
    "units": "Radians per second",
    "references": [
        "Figure 2, New J. Phys. 11 013017",
        "Table 4, Physics Reports 528 1–45"
    ]
}

g_factor_excited = 2.01
r"""
g_e

References
----------
Figure 2, New J. Phys. 11 013017
Section 3.3, Physics Reports 528 1–45
"""

nv_parameters["g_factor_excited"] = {
    "value": g_factor_excited,
    "units": "1",
    "references": [
        "Figure 2, New J. Phys. 11 013017",
        "Section 3.3, Physics Reports 528 1–45"
    ]
}

gyromagnetic_ratio_excited = g_factor_excited*bohr_magneton_gyro
r"""
\gamma_e

Units
-----
Radians per second
"""

nv_parameters["gyromagnetic_ratio_excited"] = {
    "value": gyromagnetic_ratio_excited,
    "units": "Radians per second"
}

spin_spin_relaxation_time_excited = 10.9e-9
r"""
T_{2, e}

Units
-----
Seconds

References
----------
Page 669, Nature Phys. 6 668-672
Section 5.3, Physics Reports 528 1–45
"""

nv_parameters["spin_spin_relaxation_time_excited"] = {
    "value": spin_spin_relaxation_time_excited,
    "units": "Seconds",
    "references": [
        "Page 669, Nature Phys. 6 668-672",
        "Section 5.3, Physics Reports 528 1–45"
    ]
}

# Intersystem -----------------------------------------------------------------

spin_conserving_decay_rate = 77e6
r"""
\Gamma_{gX\gets eX}

Units
-----
Per second

Reference
---------
Table 1.1, Santiago Hernandez Gomez, PhD Thesis, Universita di Firenze
"""

nv_parameters["spin_conserving_decay_rate"] = {
    "value": spin_conserving_decay_rate,
    "units": "Per second",
    "references": [
        "Table 1.1, Santiago Hernandez Gomez, "
        "PhD Thesis, Universita di Firenze"
    ]
}

radiative_mixing_angle = 0.193
r"""
\theta

Units
-----
Radians

Reference
---------
Table 1.1, Santiago Hernandez Gomez, PhD Thesis, Universita di Firenze
"""

nv_parameters["radiative_mixing_angle"] = {
    "value": radiative_mixing_angle,
    "units": "Radians",
    "references": [
        "Table 1.1, Santiago Hernandez Gomez, "
        "PhD Thesis, Universita di Firenze"
    ]
}

spin_nonconserving_decay_rate = \
    spin_conserving_decay_rate*math.tan(radiative_mixing_angle)**2
r"""
\Gamma_{gX\gets eX}

Units
-----
Per second

Reference
---------
Section 1.4.1, Santiago Hernandez Gomez, PhD Thesis, Universita di Firenze
"""

nv_parameters["spin_nonconserving_decay_rate"] = {
    "value": spin_nonconserving_decay_rate,
    "units": "Per second",
    "references": [
        "Section 1.4.1, Santiago Hernandez Gomez, "
        "PhD Thesis, Universita di Firenze"
    ]
}

pm_to_singlet_decay_rate = 60.4e6
r"""
\Gamma_{s\gets\pm}

Units
-----
Per second

Reference
---------
Table 1.1, Santiago Hernandez Gomez, PhD Thesis, Universita di Firenze
"""

nv_parameters["pm_to_singlet_decay_rate"] = {
    "value": pm_to_singlet_decay_rate,
    "units": "Per second",
    "references": [
        "Table 1.1, Santiago Hernandez Gomez, "
        "PhD Thesis, Universita di Firenze"
    ]
}

z_to_singlet_decay_rate = 9.39e6
r"""
\Gamma_{s\gets0}

Units
-----
Per second

Reference
---------
Table 1.1, Santiago Hernandez Gomez, PhD Thesis, Universita di Firenze
"""

nv_parameters["z_to_singlet_decay_rate"] = {
    "value": z_to_singlet_decay_rate,
    "units": "Per second",
    "references": [
        "Table 1.1, Santiago Hernandez Gomez, "
        "PhD Thesis, Universita di Firenze"
    ]
}

singlet_to_z_decay_rate = 9.6e6
r"""
\Gamma_{0\gets s}

Units
-----
Per second

Reference
---------
Table 1.1, Santiago Hernandez Gomez, PhD Thesis, Universita di Firenze
"""

nv_parameters["singlet_to_z_decay_rate"] = {
    "value": singlet_to_z_decay_rate,
    "units": "Per second",
    "references": [
        "Table 1.1, Santiago Hernandez Gomez, "
        "PhD Thesis, Universita di Firenze"
    ]
}

if __name__ == "__main__":
    from pogger import Pogger as Logger

    with Logger("superspinsim-generate") as logger:
        @logger.record(("nv_parameters"))
        def write_values():
            return nv_parameters

        write_values()
