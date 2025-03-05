from pogger import Read, Pogger as Logger

import numpy as np
import scipy.optimize as spo

import matplotlib.pyplot as plt
from cmcrameri import cm


def linear_template(x, a, b):
    return a*x + b


def hyperbolic_template(x, a, b, x0, y0):
    return np.sqrt((a*(x - x0))**2 + y0**2) + b


smooth_resolution = 200


if __name__ == "__main__":
    with Logger("superspinsim-benchmarks", verbose=True) as logger:

        def read_from_archive(trials):
            system_info = None
            for trial, trial_data in trials.items():
                profile_log = trial_data["profile_log"]
                read = Read("superspinsim-benchmarks", profile_log)

                if system_info is None:
                    system_info = {"cpu": {}, "gpu": {}}

                    for key in ["os", "hostname"]:
                        system_info[key] = \
                            read.read_value(key, "errors/system_info")

                    for key in ["name", "cores", "clock", "clock_units"]:
                        system_info["cpu"][key] = \
                            read.read_value(key, "errors/system_info/cpu")

                    for key in [
                            "name", "chip", "clock", "clock_units",
                            "memory", "memory_units",
                            "streaming_multiprocessors"
                            ]:
                        system_info["gpu"][key] = \
                            read.read_value(key, "errors/system_info/gpu")

                simulation_log = read.read_value("previous_log")

                errors = read.read_array("error", "errors")
                durations, durations_units = read.read_array(
                    "durations", "errors")

                trial_data["simulation_log"] = simulation_log
                trial_data["errors"] = errors
                trial_data["durations"] = durations*1e-9

                read = Read("superspinsim-benchmarks", simulation_log)
                divisions = np.floor(
                    read.read_array("divisions", "error_analysis"))

                # sample_rate = divisions/12e-6
                sample_rate = read.read_array("sample_rate", "error_analysis")

                trial_data["divisions"] = divisions
                trial_data["sample_rate"] = sample_rate

            return trials, system_info

        def fit(trials):
            for trial, trial_data in trials.items():
                fits = {}

                errors = trial_data["errors"]
                durations = trial_data["durations"]
                sample_rate = trial_data["sample_rate"]

                log_errors = np.log10(errors)
                log_durations = np.log10(durations)
                log_sample_rate = np.log10(sample_rate)

                mask = log_errors > -12
                # mask = np.ones_like(log_errors, dtype=np.bool)
                log_errors = log_errors[mask]
                log_durations = log_durations[mask]
                log_sample_rate = log_sample_rate[mask]

                try:
                    fit_params_durations, fit_covs_durations = spo.curve_fit(
                        hyperbolic_template,
                        log_errors,
                        log_durations,
                        (0.5, 1, 1, 1)
                    )

                    errors_smooth = np.geomspace(
                        errors[0], errors[mask][-1], smooth_resolution)
                    durations_fit = np.power(10, hyperbolic_template(
                        np.log10(errors_smooth),
                        *fit_params_durations
                    ))
                    kind = "hyperbolic"

                except Exception:
                    fit_params_durations, fit_covs_durations = spo.curve_fit(
                        linear_template,
                        log_errors,
                        log_durations,
                        (0.5, 1)
                    )

                    errors_smooth = np.geomspace(
                        errors[0], errors[mask][-1], smooth_resolution)
                    durations_fit = np.power(10, linear_template(
                        np.log10(errors_smooth),
                        *fit_params_durations
                    ))
                    kind = "linear"

                durations_vs_errors_fit = {
                    "kind": kind,
                    "params": fit_params_durations,
                    "covs": fit_covs_durations,
                    "power": np.abs(1/fit_params_durations[0]),
                    "errors_smooth": errors_smooth,
                    "durations_fit": durations_fit
                }
                fits["durations_vs_errors"] = durations_vs_errors_fit

                fit_params_sample_rate, fit_covs_sample_rate = spo.curve_fit(
                    linear_template,
                    log_sample_rate,
                    log_errors,
                    (2, 1)
                )
                sample_rate_smooth = np.geomspace(
                    sample_rate[0], sample_rate[mask][-1], smooth_resolution)
                errors_fit = np.power(10, linear_template(
                    np.log10(sample_rate_smooth),
                    *fit_params_sample_rate
                ))

                sample_rate_vs_errors_fit = {
                    "kind": "linear",
                    "params": fit_params_sample_rate,
                    "covs": fit_covs_sample_rate,
                    "power": np.abs(fit_params_sample_rate[0]),
                    "sample_rate_smooth": sample_rate_smooth,
                    "errors_fit": errors_fit
                }
                fits["sample_rate_vs_errors"] = sample_rate_vs_errors_fit

                trial_data["fits"] = fits

            return trials

        @logger.record(("trials", "system_info"))
        def plot(trials, system_info):
            try:
                plt.figure("comparison_gpu_time")
                for trial_index, (trial, trial_data) \
                        in enumerate(trials.items()):

                    fit = trial_data["fits"]["durations_vs_errors"]
                    durations_fit = fit["durations_fit"]
                    errors_smooth = fit["errors_smooth"]
                    power = fit["power"]

                    plt.loglog(
                        trial_data["durations"],
                        trial_data["errors"],
                        trial_data["marker"],
                        color=cm.hawaii(trial_index/len(trials)),
                        label=f"{trial.replace('_', ':')}, power={power:.2f}"
                    )

                    plt.loglog(
                        durations_fit,
                        errors_smooth,
                        "-",
                        color=cm.hawaii(trial_index/len(trials))
                    )
                plt.legend()
                plt.xlabel("GPU time (s)")
                plt.ylabel("RMS error")
                plt.gca().spines[["right", "top"]].set_visible(False)
                plt.draw()

                plt.figure("comparison_sample_rate")
                for trial_index, (trial, trial_data) \
                        in enumerate(trials.items()):

                    fit = trial_data["fits"]["sample_rate_vs_errors"]
                    sample_rate_smooth = fit["sample_rate_smooth"]
                    errors_fit = fit["errors_fit"]
                    power = fit["power"]

                    plt.loglog(
                        trial_data["sample_rate"]/1e9,
                        trial_data["errors"],
                        trial_data["marker"],
                        color=cm.hawaii(trial_index/len(trials)),
                        label=f"{trial.replace('_', ':')}, power={power:.2f}"
                    )

                    plt.loglog(
                        sample_rate_smooth/1e9,
                        errors_fit,
                        "-",
                        color=cm.hawaii(trial_index/len(trials))
                    )
                plt.legend()
                plt.xlabel("Integration stepping rate (GS/s)")
                plt.ylabel("RMS error")
                plt.gca().spines[["right", "top"]].set_visible(False)
                plt.draw()
            except Exception as exception:
                print(exception)
            finally:
                return trials, system_info

        # trials = {
        #     "CF2_1": {
        #         # "profile_log": "2025-02-21T15-57-21"
        #         # "profile_log": "2025-02-24T19-45-55"
        #         "profile_log": "2025-03-03T14-08-43",
        #         "marker": "."
        #     },

        #     "CF4_2": {
        #         # "profile_log": "2025-02-21T15-57-21"
        #         # "profile_log": "2025-02-25T19-00-58"
        #         "profile_log": "2025-02-28T19-14-51",
        #         "marker": "o"
        #     },

        #     "CF4_3": {
        #         # "profile_log": "2025-02-21T15-57-21"
        #         # "profile_log": "2025-02-25T19-00-58"
        #         "profile_log": "2025-03-03T16-01-44",
        #         "marker": ">"
        #     },

        #     "CF6_5": {
        #         "profile_log": "2025-03-04T16-51-09",
        #         "marker": "p"
        #     },

        #     "CF6_6": {
        #         "profile_log": "2025-03-04T17-01-16",
        #         "marker": "h"
        #     }
        # }

        markers = {
            "CF2_1": ".",
            "CF4_2": "o",
            "CF4_3": ">",
            "CF6_5": "p",
            "CF6_6": "h"
        }

        trials = {}
        with open("profile/profile_list.csv", "r") as file_profile_list:
            for line in file_profile_list.readlines():
                integration_method, profile_log = line.split(",")
                integration_method = integration_method.strip()
                profile_log = profile_log.strip()
                trials[integration_method] = {"profile_log": profile_log}
                trials[integration_method]["marker"] = \
                    markers[integration_method]

        _, system_info = read_from_archive(trials)
        fit(trials)
        plot(trials, system_info)
