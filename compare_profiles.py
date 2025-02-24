from pogger import Read, Pogger as Logger

import matplotlib.pyplot as plt
from cmcrameri import cm

if __name__ == "__main__":
    with Logger("superspinsim-benchmarks", verbose=True) as logger:

        @logger.record(("trials"))
        def plot(trials):
            for trial, trial_data in trials.items():
                profile_log = trial_data["profile_log"]
                read = Read("superspinsim-benchmarks", profile_log)
                simulation_log = read.read_value("previous_log")

                errors = read.read_array("error", "errors")
                durations, durations_units = read.read_array(
                    "durations", "errors")

                trial_data["simulation_log"] = simulation_log
                trial_data["errors"] = errors
                trial_data["durations"] = durations

            plt.figure("comparison")
            for trial_index, (trial, trial_data) in enumerate(trials.items()):
                plt.loglog(
                    trial_data["durations"]*1e-9,
                    trial_data["errors"],
                    ".--",
                    color=cm.hawaii(trial_index/len(trials)),
                    label=trial
                )
            plt.legend()
            plt.xlabel("GPU time (s)")
            plt.ylabel("RMS error")
            plt.show()

            return trials

        trials = {
            "CF1_2": {
                "profile_log": "2025-02-21T15-57-21"
            },
            "CF4_2": {
                "profile_log": "2025-02-21T16-15-35"
            }
        }

        plot(trials)
