import numpy as np
import math

meta_datatype = np.float128

# Gauss Legendre sampling -----------------------------------------------------

# Table 1 Alvermann and Fehske, Journal of Computational Physics
#   230 (2011) 5930-5956

_sample_gl_1 = np.array([1/2], dtype=meta_datatype)

_sample_gl_2 = np.array([1/2 - math.sqrt(3)/6, 1/2 + math.sqrt(3)/6],
                        dtype=meta_datatype)

_sample_gl_3 = np.array([1/2 - math.sqrt(3/5)/2, 1/2, 1/2 + math.sqrt(3/5)/2],
                        dtype=meta_datatype)

_sample = {
    "1_gl": _sample_gl_1,
    "2_gl": _sample_gl_2,
    "3_gl": _sample_gl_3,
}

print(f"GL1:\n{_sample_gl_1}")
print(f"GL2:\n{_sample_gl_2}")
print(f"GL3:\n{_sample_gl_3}")
print("\n")


# Q matrices ------------------------------------------------------------------

_Q_1_1_gl = np.array([[1]], dtype=meta_datatype)

# Eq (22) Blanes and Moan, Applied Numerical Mathematics 56 (2006) 1519-1537
_Q_2_2_gl = np.array(
    [
        [1/2, 1/2],
        [-math.sqrt(3)/12, math.sqrt(3)/12]
    ],
    dtype=meta_datatype
)

# Eq (22) Blanes and Moan, Applied Numerical Mathematics 56 (2006) 1519-1537
_Q_3_3_gl = np.array(
    [
        [5/18, 4/9, 5/18],
        [-math.sqrt(15)/36, 0, math.sqrt(15)/36],
        [1/24, 0, 1/24]
    ],
    dtype=meta_datatype
)


# R matrices ------------------------------------------------------------------

_R_1 = np.array([[1]], dtype=meta_datatype)

# Eq (24) Blanes and Moan, Applied Numerical Mathematics 56 (2006) 1519-1537
_R_2 = np.array(
    [
        [1, 0],
        [0, 12]
    ],
    dtype=meta_datatype
)

# Eq (24) Blanes and Moan, Applied Numerical Mathematics 56 (2006) 1519-1537
# Eq (249) Blanes, Physics Reports 470 (2009) 151–238
_R_3 = np.array(
    [
        [9/4, 0, -15],
        [0, 12, 0],
        [-15, 0, 180]
    ],
    dtype=meta_datatype
)


# X matrices ------------------------------------------------------------------


def _fill_symetric_X(x):
    """
    Eq (33) Blanes and Moan, Applied Numerical Mathematics 56 (2006) 1519-1537
    """

    number_of_exponentials = x.shape[0]
    number_of_exponentials_fill = number_of_exponentials//2
    number_of_samples = x.shape[1]
    for exponential_index in range(number_of_exponentials_fill):
        for sample_index in range(number_of_samples):
            x[number_of_exponentials - exponential_index - 1, sample_index] = \
                ((-1)**(sample_index))*x[exponential_index, sample_index]


_X_2_1 = np.array([[1]], dtype=meta_datatype)

# Eq (37) Blanes and Moan, Applied Numerical Mathematics 56 (2006) 1519-1537
_X_4_2 = np.array(
    [
        [1/2, 1/6],
        [0, 0]
    ],
    dtype=meta_datatype
)
_fill_symetric_X(_X_4_2)

# Eq (37) Blanes and Moan, Applied Numerical Mathematics 56 (2006) 1519-1537
_X_4_3 = np.array(
    [
        [0, 1/12],
        [1, 0],
        [0, 0]
    ],
    dtype=meta_datatype
)
_fill_symetric_X(_X_4_3)

# Table 1 Blanes and Moan, Applied Numerical Mathematics 56 (2006) 1519-1537
_X_6_5 = np.array(
    [
        [0.2, 0.08734395950888931101, 0.03734395950888931101],
        [0.34815492558797391479, 0.053438272547684150, 0.00584269157837031012],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ],
    dtype=meta_datatype
)
_X_6_5[2, 0] = 1 - 2*(_X_6_5[0, 0] + _X_6_5[1, 0])
_X_6_5[2, 2] = 1/12 - 2*(_X_6_5[0, 2] + _X_6_5[1, 2])
_fill_symetric_X(_X_6_5)


# Table 1 Blanes and Moan, Applied Numerical Mathematics 56 (2006) 1519-1537
_X_6_6 = np.array(
    [
        [0.208, 0.09023186422416794596, 0.03823186422416794596],
        [0.312, 0.04467385661651479788, 0.00439421553992544024],
        [0, 0.01407960659498524468, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ],
    dtype=meta_datatype
)
_X_6_6[2, 0] = 1/2 - (_X_6_6[0, 0] + _X_6_6[1, 0])
_X_6_6[2, 2] = 1/24 - (_X_6_6[0, 2] + _X_6_6[1, 2])
_fill_symetric_X(_X_6_6)


print(f"X42:\n{_X_4_2}")
print(f"X43:\n{_X_4_3}")
print(f"X65:\n{_X_6_5}")
print(f"X66:\n{_X_6_6}")
print("\n")


# rho matrices ----------------------------------------------------------------

# Eq (50) Blanes and Moan, Applied Numerical Mathematics 56 (2006) 1519-1537
_rho_2_1_gl = _X_2_1@_R_1@_Q_1_1_gl
_rho_4_2_gl = _X_4_2@_R_2@_Q_2_2_gl
_rho_4_3_gl = _X_4_3@_R_2@_Q_2_2_gl
_rho_6_5_gl = _X_6_5@_R_3@_Q_3_3_gl
_rho_6_6_gl = _X_6_6@_R_3@_Q_3_3_gl

# # Table 3 Alvermann and Fehske, Journal of Computational Physics
# #   230 (2011) 5930-5956
# _rho_6_5_gl = np.array(
#     [
#         [0.16,  0.14587456942714338561, 0.11762370828143015682],
#         [0.38752405202531186588, 0.15089113704380764664, -0.12805075909013044594],
#         [0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]
#     ],
#     dtype=meta_datatype
# )
# _rho_6_5_gl[2, 0] = 1 - 2*_rho_6_5_gl[1, 0] - 2*_rho_6_5_gl[0, 0]
# _rho_6_5_gl[2, 2] = -2*_rho_6_5_gl[1, 2] - 2*_rho_6_5_gl[0, 2]
# _fill_symetric_X(_rho_6_5_gl)
# 
# # Table 3 Alvermann and Fehske, Journal of Computational Physics
# #   230 (2011) 5930-5956
# _rho_6_6_gl = np.array(
#     [
#         [0.16, 0.15101538937746543493, 0.13304616813239630479],
#         [-0.22738164742696330169, -0.087654259755115431662, 0.087654259755115431662],
#         [0, 0.21035154512209824847, 0],
#         [0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]
#     ],
#     dtype=meta_datatype
# )
# _rho_6_6_gl[2, 0] = 1/2 - _rho_6_6_gl[1, 0] - _rho_6_6_gl[0, 0]
# _rho_6_6_gl[2, 2] = -_rho_6_6_gl[1, 2] - _rho_6_6_gl[0, 2]
# _fill_symetric_X(_rho_6_6_gl)
# 
_rho = {
    "2_1_gl": _rho_2_1_gl,
    "4_2_gl": _rho_4_2_gl,
    "4_3_gl": _rho_4_3_gl,
    "6_5_gl": _rho_6_5_gl,
    "6_6_gl": _rho_6_6_gl
}

for key, rho in _rho.items():
    print(np.sum(rho))
    # _rho[key] = rho/np.sum(rho)

print(f"rho21:\n{_rho['2_1_gl']}")
print(f"rho42:\n{_rho['4_2_gl']}")
print(f"rho43:\n{_rho['4_3_gl']}")
print(f"rho65:\n{_rho['6_5_gl']}")
print(f"rho66:\n{_rho['6_6_gl']}")


# Write -----------------------------------------------------------------------

def _write_script(samples, weights):
    with open("quadratures.py", "w") as file:
        file.write("\"\"\"\nScript generated by "
                   "`generate_quadratures.py`.\n\"\"\"\n\n\n")
        file.write("import numpy as np\n\n\n")

        file.write("samples = {}\n\n")

        for label, sample in samples.items():
            file.write(f"samples[\"{label}\"] = np.array([")
            for time in sample:
                file.write(f"{time}, ")
            file.write("], dtype=np.float128)\n\n")

        file.write("weights = {}\n\n")

        for label, weight in weights.items():
            file.write(f"weights[\"{label}\"] = np.array([\n")
            for y_index in range(weight.shape[0]):
                file.write("[")
                for x_index in range(weight.shape[1]):
                    file.write(f"{weight[y_index, x_index]}, ")
                file.write("],\n")
            file.write("], dtype=np.float128)\n\n")


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    from pogger import Pogger as Logger

    with Logger("superspinsim-generate") as logger:
        @logger.record(("samples", "weights"))
        def _visualise(samples, weights):
            from util import colour_complex_matrix as _colour_complex_matrix

            plt.figure("nodes", figsize=(6, 4))
            for plot_index, (label, sample) in enumerate(samples.items()):
                plt.subplot(4, 1, plot_index + 1)
                plt.plot(sample, [0]*sample.size, "k.")
                plt.xlim(0, 1)
                plt.text(0 - 0.05, 0, f"{plot_index + 1}-point")
                plt.axis("off")
            plt.subplot(4, 1, 4)
            plt.plot([0, 1], [0, 0], "k-")
            plt.xlim(0, 1)
            plt.text(0 - 0.05, 0, "0")
            plt.text(1 + 0.05, 0, "1")
            plt.axis("off")
            plt.draw()

            plt.figure("weights", figsize=(6, 8))
            plt.suptitle("Commutator-free weights")

            for plot_index, (label, weight) in enumerate(weights.items()):
                if plot_index == 0:
                    continue
                plt.subplot(2, 2, plot_index)
                coloured = np.array(
                    _colour_complex_matrix(weight), dtype=np.float64)
                plt.imshow(coloured)
                plt.title(label)
                plt.xticks(range(weight.shape[1]))
                plt.yticks(range(weight.shape[0]))
                if plot_index == 3:
                    plt.xlabel("GL sample")
                    plt.ylabel("Exponential\n\n")
            plt.draw()

            return samples, weights

        _write_script(_sample, _rho)
        _visualise(_sample, _rho)
        plt.show()
