import os
import subprocess


def main():
    os.chdir("tests")

    out = subprocess.check_output(["ls"])
    out = out.decode("utf-8")
    out = out.split()
    files = []
    for name in out:
        if ".py" in name:
            print(name)
            files.append(name)

    os.chdir("..")


if __name__ == "__main__":
    from pogger import Pogger as Logger
    with Logger("superspinsim-test") as logger:
        wrap = logger.record((), ())(main)
        wrap()
