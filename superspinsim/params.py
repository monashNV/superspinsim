import math


class fundamental:
    planck = 6.62607015e-34
    """
    Value:
    ------
    6.62607015 10-34J/Hz

    Reference:
    ----------
    Tiesinga, Rev. Mod. Phys. 93, 2, 025010 (2021),
    https://link.aps.org/doi/10.1103/RevModPhys.93.025010

    Note
    ----
    Value given without metric prefix.
    """

    planck_uncertainty = 0.0
    planck_units = "J/Hz"

    boltzmann = 1.3806489999999998e-23
    """
    Value:
    ------
    13.8064899 yJ/K

    Reference:
    ----------
    Tiesinga, Rev. Mod. Phys. 93, 2, 025010 (2021),
    https://link.aps.org/doi/10.1103/RevModPhys.93.025010

    Note
    ----
    Value given without metric prefix.
    """

    boltzmann_uncertainty = 0.0
    boltzmann_units = "J/K"

    bohr_gyro = 13996244917.1
    """
    Value:
    ------
    13.996244917(4) GHz/T

    Reference:
    ----------
    Mohr, Rev. Mod. Phys. 97, 2, 025002 (2025),
    https://link.aps.org/doi/10.1103/RevModPhys.97.025002

    Note
    ----
    Value given without metric prefix.
    """

    bohr_gyro_uncertainty = 4.3999999999999995
    bohr_gyro_units = "Hz/T"

    nuclear_gyro = 7622593.2188
    """
    Value:
    ------
    7.622593218(2) MHz/T

    Reference:
    ----------
    Mohr, Rev. Mod. Phys. 97, 2, 025002 (2025),
    https://link.aps.org/doi/10.1103/RevModPhys.97.025002

    Note
    ----
    Value given without metric prefix.
    """

    nuclear_gyro_uncertainty = 0.0024
    nuclear_gyro_units = "Hz/T"

    boltzmann_gyro = 20836619120.0
    """
    Value:
    ------
    20.8366191 GHz/K

    Reference:
    ----------
    Tiesinga, Rev. Mod. Phys. 93, 2, 025010 (2021),
    https://link.aps.org/doi/10.1103/RevModPhys.93.025010

    Note
    ----
    Value given without metric prefix.
    """

    boltzmann_gyro_uncertainty = 0.0
    boltzmann_gyro_units = "Hz/K"


class standards:
    class lab:
        class ntp:
            temperature = 293.15
            """
            Value:
            ------
            293.149999 K

            Reference:
            ----------
            Doiron (2007),
            https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=823211
            """

            temperature_uncertainty = 0.0
            temperature_units = "K"

            pressure = 101325.0
            """
            Value:
            ------
            101.325000 kPa

            Reference:
            ----------
            Doiron (2007),
            https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=823211

            Note
            ----
            Value given without metric prefix.
            """

            pressure_uncertainty = 0.0
            pressure_units = "Pa"


class local:
    class monash:
        class nh:
            gravity = 0.00979976
            """
            Value:
            ------
            9.79975(7) m/s^2

            Reference:
            ----------
            Tritt, PhD Thesis, Monash University (2025),
            https://bridges.monash.edu/articles/thesis/Efficiently_estimating_waveforms_on_a_compressive_ultracold_atom_sensor/28300865

            Note
            ----
            Value given without metric prefix.
            """

            gravity_uncertainty = 6.999999999999999e-08
            gravity_units = "/s^2"

            latitude = -37.90812876351057
            """
            Value:
            ------
            -37.9081287 deg

            Reference:
            ----------
            Tritt, PhD Thesis, Monash University (2025),
            https://bridges.monash.edu/articles/thesis/Efficiently_estimating_waveforms_on_a_compressive_ultracold_atom_sensor/28300865
            """

            latitude_uncertainty = 0.0
            latitude_units = "deg"

            longitude = 145.1326077738925
            """
            Value:
            ------
            145.132607 deg

            Reference:
            ----------
            Tritt, PhD Thesis, Monash University (2025),
            https://bridges.monash.edu/articles/thesis/Efficiently_estimating_waveforms_on_a_compressive_ultracold_atom_sensor/28300865
            """

            longitude_uncertainty = 0.0
            longitude_units = "deg"


class nv:
    class room:
        class optical:
            conserving = 66000000.0
            """
            Value:
            ------
            66(1) MHz

            Reference:
            ----------
            Busaite, Phys. Rev. B 102, 22, 224101 (2020),
            https://link.aps.org/doi/10.1103/PhysRevB.102.224101

            Note
            ----
            Value given without metric prefix.
            """

            conserving_uncertainty = 1000000.0
            conserving_units = "Hz"

            mixing = 11.058085446024888
            """
            Value:
            ------
            11.0(6) deg

            Reference:
            ----------
            Gomez, PhD Thesis, Universita di Firenze (2021),
            https://hdl.handle.net/2158/1234493
            """

            mixing_uncertainty = 0.6302535746439055
            mixing_units = "deg"

            nonconserving = 3547020.0564265894
            """
            Value:
            ------
            0(20) MHz

            Reference:
            ----------
            Wolters, Phys. Rev. A 88, 2, 020101 (2013),
            https://link.aps.org/doi/10.1103/PhysRevA.88.020101

            Note
            ----
            Value given without metric prefix.
            """

            nonconserving_uncertainty = 23668639.053254437
            nonconserving_units = "Hz"

            ratio = 0.01
            """
            Value:
            ------
            0.01(1)

            Reference:
            ----------
            Robledo, New Journal of Physics 13, 025013 (2011),
            https://dx.doi.org/10.1088/1367-2630/13/2/025013
            """

            ratio_uncertainty = 0.01

            prop_optical = 0.86
            """
            Value:
            ------
            0.85(2)

            Reference:
            ----------
            Robledo, New Journal of Physics 13, 025013 (2011),
            https://dx.doi.org/10.1088/1367-2630/13/2/025013
            """

            prop_optical_uncertainty = 0.02

            life_total = 1.3e-08
            """
            Value:
            ------
            13(4) ns

            Reference:
            ----------
            Wolters, Phys. Rev. A 88, 2, 020101 (2013),
            https://link.aps.org/doi/10.1103/PhysRevA.88.020101

            Note
            ----
            Value given without metric prefix.
            """

            life_total_uncertainty = 4e-09
            life_total_units = "s"

            total = 76923076.92307691
            """
            Value:
            ------
            70(20) MHz

            Reference:
            ----------
            Wolters, Phys. Rev. A 88, 2, 020101 (2013),
            https://link.aps.org/doi/10.1103/PhysRevA.88.020101

            Note
            ----
            Value given without metric prefix.
            """

            total_uncertainty = 23668639.053254437
            total_units = "Hz"

            life_conserving = 1.5151515151515152e-08
            """
            Value:
            ------
            15.1(2) ns

            Reference:
            ----------
            Busaite, Phys. Rev. B 102, 22, 224101 (2020),
            https://link.aps.org/doi/10.1103/PhysRevB.102.224101

            Note
            ----
            Value given without metric prefix.
            """

            life_conserving_uncertainty = 2.295684113865932e-10
            life_conserving_units = "s"

            prop_conserving = 0.9538887392664543
            """
            Value:
            ------
            0.9(3)

            Reference:
            ----------
            Wolters, Phys. Rev. A 88, 2, 020101 (2013),
            https://link.aps.org/doi/10.1103/PhysRevA.88.020101
            """

            prop_conserving_uncertainty = 0.30516460283887475

            prop_nonconserving = 0.04611126073354567
            """
            Value:
            ------
            0.0(3)

            Reference:
            ----------
            Wolters, Phys. Rev. A 88, 2, 020101 (2013),
            https://link.aps.org/doi/10.1103/PhysRevA.88.020101
            """

            prop_nonconserving_uncertainty = 0.3080192491282141

        class ground:
            thermalisation_time = 0.01
            """
            Value:
            ------
            10(1) ms

            Reference:
            ----------
            Busaite, Phys. Rev. B 102, 22, 224101 (2020),
            https://link.aps.org/doi/10.1103/PhysRevB.102.224101

            Note
            ----
            Value given without metric prefix.
            """

            thermalisation_time_uncertainty = 0.001
            thermalisation_time_units = "s"

            dephasing_time = 9.999999999999999e-05
            """
            Value:
            ------
            100(10) us

            Reference:
            ----------
            Busaite, Phys. Rev. B 102, 22, 224101 (2020),
            https://link.aps.org/doi/10.1103/PhysRevB.102.224101

            Note
            ----
            Value given without metric prefix.
            """

            dephasing_time_uncertainty = 9.999999999999999e-06
            dephasing_time_units = "s"

            thermalisation_time_14N = 10.0
            """
            Value:
            ------
            10(1) s

            Reference:
            ----------
            Busaite, Phys. Rev. B 102, 22, 224101 (2020),
            https://link.aps.org/doi/10.1103/PhysRevB.102.224101
            """

            thermalisation_time_14N_uncertainty = 1.0
            thermalisation_time_14N_units = "s"

            dephasing_time_14N = 9.999999999999999e-06
            """
            Value:
            ------
            10(1) us

            Reference:
            ----------
            Busaite, Phys. Rev. B 102, 22, 224101 (2020),
            https://link.aps.org/doi/10.1103/PhysRevB.102.224101

            Note
            ----
            Value given without metric prefix.
            """

            dephasing_time_14N_uncertainty = 1e-06
            dephasing_time_14N_units = "s"

            gyro_longitudinal = 28000000000.0
            """
            Value:
            ------
            28(1) GHz/T

            Reference:
            ----------
            Smeltzer, Phys. Rev. A 80, 5, 050302 (2009),
            https://link.aps.org/doi/10.1103/PhysRevA.80.050302

            Note
            ----
            Value given without metric prefix.
            """

            gyro_longitudinal_uncertainty = 1000000000.0
            gyro_longitudinal_units = "Hz/T"

            zfs_longitudinal = 2872000000.0
            """
            Value:
            ------
            2.871(2) GHz

            Reference:
            ----------
            Felton, Phys. Rev. B 79, 7, 075203 (2009),
            https://link.aps.org/doi/10.1103/PhysRevB.79.075203

            Note
            ----
            Value given without metric prefix.
            """

            zfs_longitudinal_uncertainty = 2000000.0
            zfs_longitudinal_units = "Hz"

            gyro_14N_longitudinal = -3076999.9999999995
            """
            Value:
            ------
            -3.076(1) MHz/T

            Reference:
            ----------
            Smeltzer, Phys. Rev. A 80, 5, 050302 (2009),
            https://link.aps.org/doi/10.1103/PhysRevA.80.050302

            Note
            ----
            Value given without metric prefix.
            """

            gyro_14N_longitudinal_uncertainty = 1000.0
            gyro_14N_longitudinal_units = "Hz/T"

            nuclear_quadrupole_14N_longitudinal = -4945000.0
            """
            Value:
            ------
            -4.945(5) MHz

            Reference:
            ----------
            Smeltzer, Phys. Rev. A 80, 5, 050302 (2009),
            https://link.aps.org/doi/10.1103/PhysRevA.80.050302

            Note
            ----
            Value given without metric prefix.
            """

            nuclear_quadrupole_14N_longitudinal_uncertainty = 5000.0
            nuclear_quadrupole_14N_longitudinal_units = "Hz"

            hyperfine_14N_longitudinal = -2162000.0
            """
            Value:
            ------
            -2.161(2) MHz

            Reference:
            ----------
            Smeltzer, Phys. Rev. A 80, 5, 050302 (2009),
            https://link.aps.org/doi/10.1103/PhysRevA.80.050302

            Note
            ----
            Value given without metric prefix.
            """

            hyperfine_14N_longitudinal_uncertainty = 2000.0
            hyperfine_14N_longitudinal_units = "Hz"

            hyperfine_14N_transverse = -2620000.0
            """
            Value:
            ------
            -2.62(5) MHz

            Reference:
            ----------
            Chen, Phys. Rev. B 92, 2, 020101 (2015),
            https://link.aps.org/doi/10.1103/PhysRevB.92.020101

            Note
            ----
            Value given without metric prefix.
            """

            hyperfine_14N_transverse_uncertainty = 50000.0
            hyperfine_14N_transverse_units = "Hz"

            g_longitudinal = 2.000536581621305
            """
            Value:
            ------
            2.00(7)

            Reference:
            ----------
            Smeltzer, Phys. Rev. A 80, 5, 050302 (2009),
            https://link.aps.org/doi/10.1103/PhysRevA.80.050302
            """

            g_longitudinal_uncertainty = 0.07144773505790375

            g_transverse = 2.0031
            """
            Value:
            ------
            2.0030(2)

            Reference:
            ----------
            Felton, Phys. Rev. B 79, 7, 075203 (2009),
            https://link.aps.org/doi/10.1103/PhysRevB.79.075203
            """

            g_transverse_uncertainty = 0.0002

            hyperfine_15N_longitudinal = 3030000.0
            """
            Value:
            ------
            3.02(3) MHz

            Reference:
            ----------
            Felton, Phys. Rev. B 79, 7, 075203 (2009),
            https://link.aps.org/doi/10.1103/PhysRevB.79.075203

            Note
            ----
            Value given without metric prefix.
            """

            hyperfine_15N_longitudinal_uncertainty = 30000.0
            hyperfine_15N_longitudinal_units = "Hz"

            hyperfine_15N_transverse = 3650000.0
            """
            Value:
            ------
            3.64(3) MHz

            Reference:
            ----------
            Felton, Phys. Rev. B 79, 7, 075203 (2009),
            https://link.aps.org/doi/10.1103/PhysRevB.79.075203

            Note
            ----
            Value given without metric prefix.
            """

            hyperfine_15N_transverse_uncertainty = 30000.0
            hyperfine_15N_transverse_units = "Hz"

            hyperfine_13C_longitudinal = 198200000.0
            """
            Value:
            ------
            198.1(3) MHz

            Reference:
            ----------
            Felton, Phys. Rev. B 79, 7, 075203 (2009),
            https://link.aps.org/doi/10.1103/PhysRevB.79.075203

            Note
            ----
            Value given without metric prefix.
            """

            hyperfine_13C_longitudinal_uncertainty = 300000.0
            hyperfine_13C_longitudinal_units = "Hz"

            hyperfine_13C_transverse = 120800000.0
            """
            Value:
            ------
            120.7(2) MHz

            Reference:
            ----------
            Felton, Phys. Rev. B 79, 7, 075203 (2009),
            https://link.aps.org/doi/10.1103/PhysRevB.79.075203

            Note
            ----
            Value given without metric prefix.
            """

            hyperfine_13C_transverse_uncertainty = 200000.0
            hyperfine_13C_transverse_units = "Hz"

            hyperfine_13C_theta = 125.26
            """
            Value:
            ------
            125.26(5) deg

            Reference:
            ----------
            Felton, Phys. Rev. B 79, 7, 075203 (2009),
            https://link.aps.org/doi/10.1103/PhysRevB.79.075203
            """

            hyperfine_13C_theta_uncertainty = 0.05
            hyperfine_13C_theta_units = "deg"

            gyro_transverse = 28035878231.501907
            """
            Value:
            ------
            28.035(3) GHz/T

            Reference:
            ----------
            Felton, Phys. Rev. B 79, 7, 075203 (2009),
            https://link.aps.org/doi/10.1103/PhysRevB.79.075203

            Note
            ----
            Value given without metric prefix.
            """

            gyro_transverse_uncertainty = 2799248.9872326427
            gyro_transverse_units = "Hz/T"

            g_14N_longitudinal = -0.00021984468077316984
            """
            Value:
            ------
            -0.00021984(7)

            Reference:
            ----------
            Smeltzer, Phys. Rev. A 80, 5, 050302 (2009),
            https://link.aps.org/doi/10.1103/PhysRevA.80.050302
            """

            g_14N_longitudinal_uncertainty = 7.144773505793423e-08

            g_N_14N_longitudinal = 0.40366839834155777
            """
            Value:
            ------
            0.4036(1)

            Reference:
            ----------
            Smeltzer, Phys. Rev. A 80, 5, 050302 (2009),
            https://link.aps.org/doi/10.1103/PhysRevA.80.050302
            """

            g_N_14N_longitudinal_uncertainty = 0.0001311889497373194

            thermalisation_rate = 100.0
            """
            Value:
            ------
            100(10) Hz

            Reference:
            ----------
            Busaite, Phys. Rev. B 102, 22, 224101 (2020),
            https://link.aps.org/doi/10.1103/PhysRevB.102.224101
            """

            thermalisation_rate_uncertainty = 10.0
            thermalisation_rate_units = "Hz"

            dephasing_rate = 10000.0
            """
            Value:
            ------
            10(1) kHz

            Reference:
            ----------
            Busaite, Phys. Rev. B 102, 22, 224101 (2020),
            https://link.aps.org/doi/10.1103/PhysRevB.102.224101

            Note
            ----
            Value given without metric prefix.
            """

            dephasing_rate_uncertainty = 1000.0000000000002
            dephasing_rate_units = "Hz"

            thermalisation_rate_14N = 0.1
            """
            Value:
            ------
            100(10) mHz

            Reference:
            ----------
            Busaite, Phys. Rev. B 102, 22, 224101 (2020),
            https://link.aps.org/doi/10.1103/PhysRevB.102.224101

            Note
            ----
            Value given without metric prefix.
            """

            thermalisation_rate_14N_uncertainty = 0.01
            thermalisation_rate_14N_units = "Hz"

            dephasing_rate_14N = 100000.00000000001
            """
            Value:
            ------
            100(10) kHz

            Reference:
            ----------
            Busaite, Phys. Rev. B 102, 22, 224101 (2020),
            https://link.aps.org/doi/10.1103/PhysRevB.102.224101

            Note
            ----
            Value given without metric prefix.
            """

            dephasing_rate_14N_uncertainty = 10000.000000000002
            dephasing_rate_14N_units = "Hz"

        class excited:
            thermalisation_time = 0.001
            """
            Value:
            ------
            1000(500) us

            Reference:
            ----------
            Busaite, Phys. Rev. B 102, 22, 224101 (2020),
            https://link.aps.org/doi/10.1103/PhysRevB.102.224101

            Note
            ----
            Value given without metric prefix.
            """

            thermalisation_time_uncertainty = 0.0005
            thermalisation_time_units = "s"

            dephasing_time = 1e-08
            """
            Value:
            ------
            10(1) ns

            Reference:
            ----------
            Busaite, Phys. Rev. B 102, 22, 224101 (2020),
            https://link.aps.org/doi/10.1103/PhysRevB.102.224101

            Note
            ----
            Value given without metric prefix.
            """

            dephasing_time_uncertainty = 1e-09
            dephasing_time_units = "s"

            thermalisation_time_14N = 0.1
            """
            Value:
            ------
            100(10) ms

            Reference:
            ----------
            Busaite, Phys. Rev. B 102, 22, 224101 (2020),
            https://link.aps.org/doi/10.1103/PhysRevB.102.224101

            Note
            ----
            Value given without metric prefix.
            """

            thermalisation_time_14N_uncertainty = 0.01
            thermalisation_time_14N_units = "s"

            dephasing_time_14N = 0.001
            """
            Value:
            ------
            1000(500) us

            Reference:
            ----------
            Busaite, Phys. Rev. B 102, 22, 224101 (2020),
            https://link.aps.org/doi/10.1103/PhysRevB.102.224101

            Note
            ----
            Value given without metric prefix.
            """

            dephasing_time_14N_uncertainty = 0.0005
            dephasing_time_14N_units = "s"

            zfs_longitudinal = 1423000000.0
            """
            Value:
            ------
            1.42(1) GHz

            Reference:
            ----------
            Neumann, New Journal of Physics 11, 013017 (2009),
            https://iopscience.iop.org/article/10.1088/1367-2630/11/1/013017

            Note
            ----
            Value given without metric prefix.
            """

            zfs_longitudinal_uncertainty = 10000000.0
            zfs_longitudinal_units = "Hz"

            g_longitudinal = 2.0019655363224635
            """
            Value:
            ------
            2.0019(7)

            Reference:
            ----------
            Poggiali, Phys. Rev. B 95, 19, 195308 (2017),
            https://link.aps.org/doi/10.1103/PhysRevB.95.195308
            """

            g_longitudinal_uncertainty = 0.0007144773505792904

            hyperfine_14N_transverse = -23000000.0
            """
            Value:
            ------
            -23(3) MHz

            Reference:
            ----------
            Poggiali, Phys. Rev. B 95, 19, 195308 (2017),
            https://link.aps.org/doi/10.1103/PhysRevB.95.195308

            Note
            ----
            Value given without metric prefix.
            """

            hyperfine_14N_transverse_uncertainty = 3000000.0
            hyperfine_14N_transverse_units = "Hz"

            nuclear_quadrupole_14N_longitudinal = -4945000.0
            """
            Value:
            ------
            -4.945(1) MHz

            Reference:
            ----------
            Poggiali, Phys. Rev. B 95, 19, 195308 (2017),
            https://link.aps.org/doi/10.1103/PhysRevB.95.195308

            Note
            ----
            Value given without metric prefix.
            """

            nuclear_quadrupole_14N_longitudinal_uncertainty = 1000.0
            nuclear_quadrupole_14N_longitudinal_units = "Hz"

            gyro_14N_longitudinal = -3080000.0
            """
            Value:
            ------
            -3.08(1) MHz/T

            Reference:
            ----------
            Poggiali, Phys. Rev. B 95, 19, 195308 (2017),
            https://link.aps.org/doi/10.1103/PhysRevB.95.195308

            Note
            ----
            Value given without metric prefix.
            """

            gyro_14N_longitudinal_uncertainty = 10000.0
            gyro_14N_longitudinal_units = "Hz/T"

            gyro_longitudinal = 28020000000.0
            """
            Value:
            ------
            28.01(1) GHz/T

            Reference:
            ----------
            Poggiali, Phys. Rev. B 95, 19, 195308 (2017),
            https://link.aps.org/doi/10.1103/PhysRevB.95.195308

            Note
            ----
            Value given without metric prefix.
            """

            gyro_longitudinal_uncertainty = 10000000.0
            gyro_longitudinal_units = "Hz/T"

            life_0 = 1.3260000000000001e-08
            """
            Value:
            ------
            13.25(3) ns

            Reference:
            ----------
            Robledo, New Journal of Physics 13, 025013 (2011),
            https://dx.doi.org/10.1088/1367-2630/13/2/025013

            Note
            ----
            Value given without metric prefix.
            """

            life_0_uncertainty = 3e-11
            life_0_units = "s"

            life_1 = 6.89e-09
            """
            Value:
            ------
            6.88(6) ns

            Reference:
            ----------
            Robledo, New Journal of Physics 13, 025013 (2011),
            https://dx.doi.org/10.1088/1367-2630/13/2/025013

            Note
            ----
            Value given without metric prefix.
            """

            life_1_uncertainty = 6e-11
            life_1_units = "s"

            hyperfine_14N_longitudinal = -40000000.0
            """
            Value:
            ------
            -40(5) MHz

            Reference:
            ----------
            Steiner, Phys. Rev. B 81, 3, 035205 (2010),
            https://link.aps.org/doi/10.1103/PhysRevB.81.035205

            Note
            ----
            Value given without metric prefix.
            """

            hyperfine_14N_longitudinal_uncertainty = 5000000.0
            hyperfine_14N_longitudinal_units = "Hz"

            g_14N_longitudinal = -0.00022005902397834358
            """
            Value:
            ------
            -0.0002200(7)

            Reference:
            ----------
            Poggiali, Phys. Rev. B 95, 19, 195308 (2017),
            https://link.aps.org/doi/10.1103/PhysRevB.95.195308
            """

            g_14N_longitudinal_uncertainty = 7.144773505790407e-07

            g_N_14N_longitudinal = 0.4040619651907696
            """
            Value:
            ------
            0.404(1)

            Reference:
            ----------
            Poggiali, Phys. Rev. B 95, 19, 195308 (2017),
            https://link.aps.org/doi/10.1103/PhysRevB.95.195308
            """

            g_N_14N_longitudinal_uncertainty = 0.0013118894973726343

            thermalisation_rate = 1000.0
            """
            Value:
            ------
            1000(500) Hz

            Reference:
            ----------
            Busaite, Phys. Rev. B 102, 22, 224101 (2020),
            https://link.aps.org/doi/10.1103/PhysRevB.102.224101
            """

            thermalisation_rate_uncertainty = 500.00000000000006
            thermalisation_rate_units = "Hz"

            dephasing_rate = 100000000.0
            """
            Value:
            ------
            100(10) MHz

            Reference:
            ----------
            Busaite, Phys. Rev. B 102, 22, 224101 (2020),
            https://link.aps.org/doi/10.1103/PhysRevB.102.224101

            Note
            ----
            Value given without metric prefix.
            """

            dephasing_rate_uncertainty = 10000000.0
            dephasing_rate_units = "Hz"

            thermalisation_rate_14N = 10.0
            """
            Value:
            ------
            10.0(10) Hz

            Reference:
            ----------
            Busaite, Phys. Rev. B 102, 22, 224101 (2020),
            https://link.aps.org/doi/10.1103/PhysRevB.102.224101
            """

            thermalisation_rate_14N_uncertainty = 0.9999999999999998
            thermalisation_rate_14N_units = "Hz"

            dephasing_rate_14N = 1000.0
            """
            Value:
            ------
            1000(500) Hz

            Reference:
            ----------
            Busaite, Phys. Rev. B 102, 22, 224101 (2020),
            https://link.aps.org/doi/10.1103/PhysRevB.102.224101
            """

            dephasing_rate_14N_uncertainty = 500.00000000000006
            dephasing_rate_14N_units = "Hz"

            rate_0 = 75414781.29713424
            """
            Value:
            ------
            75.4(2) MHz

            Reference:
            ----------
            Robledo, New Journal of Physics 13, 025013 (2011),
            https://dx.doi.org/10.1088/1367-2630/13/2/025013

            Note
            ----
            Value given without metric prefix.
            """

            rate_0_uncertainty = 170621.6771428376
            rate_0_units = "Hz"

            rate_1 = 145137880.98693758
            """
            Value:
            ------
            145(1) MHz

            Reference:
            ----------
            Robledo, New Journal of Physics 13, 025013 (2011),
            https://dx.doi.org/10.1088/1367-2630/13/2/025013

            Note
            ----
            Value given without metric prefix.
            """

            rate_1_uncertainty = 1263900.2698427076
            rate_1_units = "Hz"

        class isc:
            s_gets_pm = 60400000.0
            """
            Value:
            ------
            60.3(3) MHz

            Reference:
            ----------
            Gomez, PhD Thesis, Universita di Firenze (2021),
            https://hdl.handle.net/2158/1234493

            Note
            ----
            Value given without metric prefix.
            """

            s_gets_pm_uncertainty = 300000.0
            s_gets_pm_units = "Hz"

            s_gets_z = 9390000.0
            """
            Value:
            ------
            9.39(5) MHz

            Reference:
            ----------
            Gomez, PhD Thesis, Universita di Firenze (2021),
            https://hdl.handle.net/2158/1234493

            Note
            ----
            Value given without metric prefix.
            """

            s_gets_z_uncertainty = 50000.0
            s_gets_z_units = "Hz"

            z_gets_s = 9600000.0
            """
            Value:
            ------
            9.59(5) MHz

            Reference:
            ----------
            Gomez, PhD Thesis, Universita di Firenze (2021),
            https://hdl.handle.net/2158/1234493

            Note
            ----
            Value given without metric prefix.
            """

            z_gets_s_uncertainty = 50000.0
            z_gets_s_units = "Hz"

            pm_gets_s = 2110000.0
            """
            Value:
            ------
            2.10(4) MHz

            Reference:
            ----------
            Gupta, J. Opt. Soc. Am. B 33, B28--B34 (2016),
            https://opg.optica.org/josab/abstract.cfm?URI=josab-33-3-B28

            Note
            ----
            Value given without metric prefix.
            """

            pm_gets_s_uncertainty = 40000.0
            pm_gets_s_units = "Hz"

            out_ratio = 1.15
            """
            Value:
            ------
            1.14(5)

            Reference:
            ----------
            Robledo, New Journal of Physics 13, 025013 (2011),
            https://dx.doi.org/10.1088/1367-2630/13/2/025013
            """

            out_ratio_uncertainty = 0.05

            life_out = 1.440922190201729e-07
            """
            Value:
            ------
            144(1) ns

            Reference:
            ----------
            Gupta, J. Opt. Soc. Am. B 33, B28--B34 (2016),
            https://opg.optica.org/josab/abstract.cfm?URI=josab-33-3-B28

            Note
            ----
            Value given without metric prefix.
            """

            life_out_uncertainty = 1.1745081865750027e-09
            life_out_units = "s"

            prop_s_gets_z = 0.14
            """
            Value:
            ------
            0.14(2)

            Reference:
            ----------
            Robledo, New Journal of Physics 13, 025013 (2011),
            https://dx.doi.org/10.1088/1367-2630/13/2/025013
            """

            prop_s_gets_z_uncertainty = 0.02

            prop_s_gets_pm = 0.55
            """
            Value:
            ------
            0.55(1)

            Reference:
            ----------
            Robledo, New Journal of Physics 13, 025013 (2011),
            https://dx.doi.org/10.1088/1367-2630/13/2/025013
            """

            prop_s_gets_pm_uncertainty = 0.01

            life_z_gets_s = 1.0416666666666667e-07
            """
            Value:
            ------
            104.1(5) ns

            Reference:
            ----------
            Gomez, PhD Thesis, Universita di Firenze (2021),
            https://hdl.handle.net/2158/1234493

            Note
            ----
            Value given without metric prefix.
            """

            life_z_gets_s_uncertainty = 5.425347222222221e-10
            life_z_gets_s_units = "s"

            mixing_in = 21.518757194298377
            """
            Value:
            ------
            21.51(7) deg

            Reference:
            ----------
            Gomez, PhD Thesis, Universita di Firenze (2021),
            https://hdl.handle.net/2158/1234493
            """

            mixing_in_uncertainty = 0.07118448599504662
            mixing_in_units = "deg"

            life_in = 1.4328700386874911e-08
            """
            Value:
            ------
            14.32(6) ns

            Reference:
            ----------
            Gomez, PhD Thesis, Universita di Firenze (2021),
            https://hdl.handle.net/2158/1234493

            Note
            ----
            Value given without metric prefix.
            """

            life_in_uncertainty = 6.244310203550102e-11
            life_in_units = "s"

            total_in = 69790000.0
            """
            Value:
            ------
            69.7(3) MHz

            Reference:
            ----------
            Gomez, PhD Thesis, Universita di Firenze (2021),
            https://hdl.handle.net/2158/1234493

            Note
            ----
            Value given without metric prefix.
            """

            total_in_uncertainty = 304138.126514911
            total_in_units = "Hz"

            mixing_out = 33.46265244806581
            """
            Value:
            ------
            33.4(3) deg

            Reference:
            ----------
            Gupta, J. Opt. Soc. Am. B 33, B28--B34 (2016),
            https://opg.optica.org/josab/abstract.cfm?URI=josab-33-3-B28
            """

            mixing_out_uncertainty = 0.2726166029732545
            mixing_out_units = "deg"

            total_out = 6940000.0
            """
            Value:
            ------
            6.94(6) MHz

            Reference:
            ----------
            Gupta, J. Opt. Soc. Am. B 33, B28--B34 (2016),
            https://opg.optica.org/josab/abstract.cfm?URI=josab-33-3-B28

            Note
            ----
            Value given without metric prefix.
            """

            total_out_uncertainty = 56568.5424949238
            total_out_units = "Hz"

            life_pm_gets_s = 4.739336492890995e-07
            """
            Value:
            ------
            473(9) ns

            Reference:
            ----------
            Gupta, J. Opt. Soc. Am. B 33, B28--B34 (2016),
            https://opg.optica.org/josab/abstract.cfm?URI=josab-33-3-B28

            Note
            ----
            Value given without metric prefix.
            """

            life_pm_gets_s_uncertainty = 8.984524157139327e-09
            life_pm_gets_s_units = "s"

            life_s_gets_z = 1.0649627263045794e-07
            """
            Value:
            ------
            106.4(6) ns

            Reference:
            ----------
            Gomez, PhD Thesis, Universita di Firenze (2021),
            https://hdl.handle.net/2158/1234493

            Note
            ----
            Value given without metric prefix.
            """

            life_s_gets_z_uncertainty = 5.670728042090412e-10
            life_s_gets_z_units = "s"

            life_s_gets_pm = 1.6556291390728476e-08
            """
            Value:
            ------
            16.55(8) ns

            Reference:
            ----------
            Gomez, PhD Thesis, Universita di Firenze (2021),
            https://hdl.handle.net/2158/1234493

            Note
            ----
            Value given without metric prefix.
            """

            life_s_gets_pm_uncertainty = 8.223323538441297e-11
            life_s_gets_pm_units = "s"

            prop_z_gets_s = 0.6959654178674352
            """
            Value:
            ------
            0.695(8)

            Reference:
            ----------
            Gupta, J. Opt. Soc. Am. B 33, B28--B34 (2016),
            https://opg.optica.org/josab/abstract.cfm?URI=josab-33-3-B28
            """

            prop_z_gets_s_uncertainty = 0.008087126417409475

            prop_pm_gets_s = 0.30403458213256485
            """
            Value:
            ------
            0.304(6)

            Reference:
            ----------
            Gupta, J. Opt. Soc. Am. B 33, B28--B34 (2016),
            https://opg.optica.org/josab/abstract.cfm?URI=josab-33-3-B28
            """

            prop_pm_gets_s_uncertainty = 0.006273885893514221


class n:
    class n_14:
        class nmr:
            gyro = -3070000.0
            """
            Value:
            ------
            -3.06(1) MHz/T

            Reference:
            ----------
            Dib, Chapter Three - Recent Advances in 14N Solid-State NMR (2016),
            https://www.sciencedirect.com/science/article/pii/S0066410315000319

            Note
            ----
            Value given without metric prefix.
            """

            gyro_uncertainty = 10000.0
            gyro_units = "Hz/T"

            g = -0.00021934454662776454
            """
            Value:
            ------
            -0.0002193(7)

            Reference:
            ----------
            Dib, Chapter Three - Recent Advances in 14N Solid-State NMR (2016),
            https://www.sciencedirect.com/science/article/pii/S0066410315000319
            """

            g_uncertainty = 7.144773505790407e-07

            g_N = 0.402750075693397
            """
            Value:
            ------
            0.402(1)

            Reference:
            ----------
            Dib, Chapter Three - Recent Advances in 14N Solid-State NMR (2016),
            https://www.sciencedirect.com/science/article/pii/S0066410315000319
            """

            g_N_uncertainty = 0.0013118894973726343


class rb:
    class rb_87:
        class physical:
            S = 0.5
            """
            Value:
            ------
            0.500000000

            Reference:
            ----------
            Steck, Rubidium 87 D Line Data (2024),
            https://steck.us/alkalidata
            """

            S_uncertainty = 0

            I = 1.5
            """
            Value:
            ------
            1.50000000

            Reference:
            ----------
            Steck, Rubidium 87 D Line Data (2024),
            https://steck.us/alkalidata
            """

            I_uncertainty = 0

            g_I = -0.0009951414
            """
            Value:
            ------
            -0.000995141(1)

            Reference:
            ----------
            Steck, Rubidium 87 D Line Data (2024),
            https://steck.us/alkalidata
            """

            g_I_uncertainty = 1e-09

            gyro_I = -13928242.780453466
            """
            Value:
            ------
            -13.92824(1) MHz/T

            Reference:
            ----------
            Steck, Rubidium 87 D Line Data (2024),
            https://steck.us/alkalidata

            Note
            ----
            Value given without metric prefix.
            """

            gyro_I_uncertainty = 13.996245560160405
            gyro_I_units = "Hz/T"

            g_N = 1.827231542053304
            """
            Value:
            ------
            1.827231(2)

            Reference:
            ----------
            Steck, Rubidium 87 D Line Data (2024),
            https://steck.us/alkalidata
            """

            g_N_uncertainty = 1.836152838077147e-06

        class hyperfine_52s12:
            hyperfine = 3417341305.452145
            """
            Value:
            ------
            3.41734130545214(4) GHz

            Reference:
            ----------
            Steck, Rubidium 87 D Line Data (2024),
            https://steck.us/alkalidata

            Note
            ----
            Value given without metric prefix.
            """

            hyperfine_uncertainty = 4.4999999999999996e-05
            hyperfine_units = "Hz"

            g_J = 2.00233107
            """
            Value:
            ------
            2.00233106(3)

            Reference:
            ----------
            Steck, Rubidium 87 D Line Data (2024),
            https://steck.us/alkalidata
            """

            g_J_uncertainty = 2.6e-08

            gyro_J = 28025116098.883194
            """
            Value:
            ------
            28.0251160(4) GHz/T

            Reference:
            ----------
            Steck, Rubidium 87 D Line Data (2024),
            https://steck.us/alkalidata

            Note
            ----
            Value given without metric prefix.
            """

            gyro_J_uncertainty = 363.99953057469054
            gyro_J_units = "Hz/T"

        class hyperfine_52p12:
            hyperfine = 407250000.0
            """
            Value:
            ------
            407.2(6) MHz

            Reference:
            ----------
            Steck, Rubidium 87 D Line Data (2024),
            https://steck.us/alkalidata

            Note
            ----
            Value given without metric prefix.
            """

            hyperfine_uncertainty = 630000.0
            hyperfine_units = "Hz"

            g_J = 0.666
            """
            Value:
            ------
            0.665(1)

            Reference:
            ----------
            Steck, Rubidium 87 D Line Data (2024),
            https://steck.us/alkalidata
            """

            g_J_uncertainty = 0.001

            gyro_J = 9321499127.4426
            """
            Value:
            ------
            9.32(1) GHz/T

            Reference:
            ----------
            Steck, Rubidium 87 D Line Data (2024),
            https://steck.us/alkalidata

            Note
            ----
            Value given without metric prefix.
            """

            gyro_J_uncertainty = 13996244.936100282
            gyro_J_units = "Hz/T"

        class hyperfine_52p32:
            hyperfine = 84718500.0
            """
            Value:
            ------
            84.718(2) MHz

            Reference:
            ----------
            Steck, Rubidium 87 D Line Data (2024),
            https://steck.us/alkalidata

            Note
            ----
            Value given without metric prefix.
            """

            hyperfine_uncertainty = 2000.0
            hyperfine_units = "Hz"

            g_J = 1.3341
            """
            Value:
            ------
            1.3341(2)

            Reference:
            ----------
            Steck, Rubidium 87 D Line Data (2024),
            https://steck.us/alkalidata
            """

            g_J_uncertainty = 0.0002

            hyperfine_quadrupole = 2496500.0
            """
            Value:
            ------
            2.496(4) MHz

            Reference:
            ----------
            Steck, Rubidium 87 D Line Data (2024),
            https://steck.us/alkalidata

            Note
            ----
            Value given without metric prefix.
            """

            hyperfine_quadrupole_uncertainty = 3700.0
            hyperfine_quadrupole_units = "Hz"

            gyro_J = 18672390369.25101
            """
            Value:
            ------
            18.672(3) GHz/T

            Reference:
            ----------
            Steck, Rubidium 87 D Line Data (2024),
            https://steck.us/alkalidata

            Note
            ----
            Value given without metric prefix.
            """

            gyro_J_uncertainty = 2799248.987225608
            gyro_J_units = "Hz/T"


