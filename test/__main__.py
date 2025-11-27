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

    print("\n")

    for version in range(11, 14):
        print("Using python version", version)
        subprocess.check_output(["uv", "python", "pin", "3." + str(version)])

        print("Running tests")
        for name in files:
            print("Running", name)
            out = subprocess.check_output(["uv", "run", name])
            out = out.decode("utf-8")
            print(out)
            print("Done\n")

    os.chdir("..")


if __name__ == "__main__":
    from pogger import Pogger as Logger
    with Logger("superspinsim-test") as logger:
        wrap = logger.record((), ())(main)
        wrap()
