import superspinsim.params.general
import superspinsim.params.nv


def write_values():
    value_writers = {
        "general": superspinsim.params.general.write_values,
        "nv": superspinsim.params.nv.write_values
    }

    from pogger import Pogger as Logger
    with Logger("superspinsim-generate") as logger:
        for key, value_writer in value_writers.items():
            value_writer_wrapped = logger.record((key))(value_writer)
            value_writer_wrapped()


if __name__ == "__main__":
    write_values()
