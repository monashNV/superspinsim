import os
import subprocess


def main():
    toml_path = "../pyproject.toml"
    with open(toml_path, "r") as toml:
        toml_old = toml.read()

    toml_expand = ""
    for line in toml_old.split("\n"):
        if "requires-python" not in line:
            toml_expand += line + "\n"
    with open(toml_path, "w") as toml:
        toml.write(toml_expand)

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

    versions = []
    results = {}
    for version in range(10, 15):
        for sub in range(0, 20):
            version_full = "3." + str(version) + "." + str(sub)
            print("Using python version", version_full)
            subprocess.check_output(["uv", "python", "pin", version_full])
            try:
                subprocess.check_output(["uv", "sync", "--reinstall"])
                versions.append(version_full)
            except subprocess.CalledProcessError as exception:
                print("Exception:", exception)
                print("Return code:", exception.returncode)
                print("Done\n")

                if exception.returncode == 1:
                    result = {}
                    results[version_full] = result
                    result["installed"] = "Fail"

                continue

            result = {}
            results[version_full] = result
            result["installed"] = "Pass"

            print("Running tests")
            for name in files:
                print("Running", name)
                try:
                    out = subprocess.check_output(["uv", "run", name])
                    out = out.decode("utf-8")
                    print(out)
                    result[name] = "Pass"
                    print("Done\n")
                except subprocess.CalledProcessError as exception:
                    print("Exception:", exception)
                    print("Return code:", exception.returncode)
                    print("Done\n")
                    result[name] = "Crash"

    for version, result in results.items():
        print(version)
        for name, outcome in result.items():
            print(f"  {name}: {outcome}")

    os.chdir("..")

    with open(toml_path, "w") as toml:
        toml.write(toml_old)


if __name__ == "__main__":
    from pogger import Pogger as Logger
    with Logger("superspinsim-test") as logger:
        wrap = logger.record((), ())(main)
        wrap()
