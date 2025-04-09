import math

parameters = {}

# Fundamental -----------------------------------------------------------------

bohr_magneton_gyro = math.tau*13.9962449171e9
r"""
The Bohr magneton divided by the reduced Plank constant.

Value
-----
2\pi \times 13.9962449171 10^9.


Units
-----
Radians per second per Tesla.


Reference
---------
Table XXXIII, 2022 CODATA recommended values
"""

parameters["bohr_magneton_gyro"] = {
    "value": bohr_magneton_gyro,
    "units": "Radians per second per Tesla",
    "references": [
        "Table XXXIII, 2022 CODATA recommended values"
    ]
}

nuclear_magneton_gyro = math.tau*7.6225932188e6
r"""
The nuclear magneton divided by the reduced Plank constant.

Value
-----
2\pi \times 7.6225932188 10^6.


Units
-----
Radians per second per Tesla.


Reference
---------
Table XXXIII, 2022 CODATA recommended values
"""

parameters["nuclear_magneton_gyro"] = {
    "value": nuclear_magneton_gyro,
    "units": "Radians per second per Tesla",
    "references": [
        "Table XXXIII, 2022 CODATA recommended values"
    ]
}

boltzmann_gyro = math.tau*20.83661912e9
r"""
The Boltzmann constant divided by the reduced Plank constant.

Value
-----
2\pi \times 20.83661912 10^9.


Units
-----
Radians per second per Kelvin.


Reference
---------
Table XXXIII, 2022 CODATA recommended values
"""

parameters["nuclear_magneton_gyro"] = {
    "value": boltzmann_gyro,
    "units": "Radians per second per Kelvin",
    "references": [
        "Table XXXIII, 2022 CODATA recommended values"
    ]
}

# Temperature -----------------------------------------------------------------

room_temperature = 293.15
r"""
Normal temperature (from normal temperature and pressure, NTP).

Value
-----
293.15


Units
-----
Kelvin
"""

parameters["room_temperature"] = {
    "value": room_temperature,
    "units": "Kelvin"
}

standard_temperature = 273.15
r"""
Standard temperature.

Value
-----
273.15


Units
-----
Kelvin
"""

parameters["standard_temperature"] = {
    "value": standard_temperature,
    "units": "Kelvin"
}


def write_values():
    return parameters

