import os
import subprocess


TOML_PATH = "../pyproject.toml"


def relax_toml():
    with open(TOML_PATH, "r") as toml:
        toml_old = toml.read()

    toml_expand = ""
    for line in toml_old.split("\n"):
        if "requires-python" not in line:
            toml_expand += line + "\n"
    with open(TOML_PATH, "w") as toml:
        toml.write(toml_expand)

    return toml_old


def revert_toml(toml_old: str):
    with open(TOML_PATH, "w") as toml:
        toml.write(toml_old)


def set_python_version(version: int, sub: int):
    version_full = f"3.{version}.{sub}"
    print("Using python version", version_full)
    subprocess.check_output(["uv", "python", "pin", version_full])
    return version_full


def list_tests():
    out = subprocess.check_output(["ls"])
    out = out.decode("utf-8")
    out = out.split()
    files = []
    for name in out:
        if ".py" in name:
            print(name)
            files.append(name)
    print("\n")
    return files


def loop_files(files: list[str], result: dict[str, str]):
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


def loop_versions(files: list[str]):
    versions = []
    results = {}
    for version in range(10, 15):
        for sub in range(0, 20):
            version_full = set_python_version(version, sub)
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
                    result["install"] = "Fail"

                continue

            result = {}
            results[version_full] = result
            result["install"] = "Pass"

            loop_files(files, result)
    return results


def print_results(results: dict):
    for version, result in results.items():
        print(version)
        for name, outcome in result.items():
            print(f"  {name}: {outcome}")


def main():
    toml_old = relax_toml()

    os.chdir("tests")

    files = list_tests()

    results = loop_versions(files)

    os.chdir("..")

    print_results(results)

    revert_toml(toml_old)

    return results


if __name__ == "__main__":
    from pogger import Pogger as Logger
    with Logger("superspinsim-test") as logger:
        wrap = logger.record(("results"), (None))(main)
        wrap()
