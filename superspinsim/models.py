import math
import numpy as np

from superspinsim import params as s3p


def nv_7(quiescent_magnetic_field: np.ndarray = None, params=s3p):
    if quiescent_magnetic_field is None:
        quiescent_magnetic_field = np.zeros(3)

    nv_ground = {
        "S": 1,
        "g": params.nv.room.ground.g_longitudinal,
        "g_perp": params.nv.room.ground.g_transverse,
        "D": math.tau*params.nv.room.ground.zfs_longitudinal,
        "TS1": params.nv.room.ground.thermalisation_time,
        "TS2": params.nv.room.ground.dephasing_time,

        "B0": quiescent_magnetic_field,
        "T": params.standards.lab.ntp.temperature
    }

    nv_excited = {
        "S": 1,
        "g": params.nv.room.excited.g_longitudinal,
        "D": math.tau*params.nv.room.excited.zfs_longitudinal,
        "TS1": params.nv.room.excited.thermalisation_time,
        "TS2": params.nv.room.excited.dephasing_time,

        "B0": quiescent_magnetic_field,
        "T": params.standards.lab.ntp.temperature
    }

    nv_singlet = {
        "S": 0,
    }

    nv_orbitals = {
        # Optical transitions
        ((0, 0), (1, 0)): {
            "rel": params.nv.room.optical.conserving,
            "rel_n": params.nv.room.optical.nonconserving
        },

        # ISC excited
        ((1, 0), (2, 0)): {
            "s_gets_0": params.nv.room.isc.s_gets_z,
            "s_gets_1": params.nv.room.isc.s_gets_pm
        },

        # ISC ground
        ((2, 0), (0, 0)): {
            "0_gets_s": params.nv.room.isc.z_gets_s,
            "1_gets_s": params.nv.room.isc.pm_gets_s
        }
    }
    return [[nv_ground], [nv_excited], [nv_singlet]], [{}, {}, {}], nv_orbitals
