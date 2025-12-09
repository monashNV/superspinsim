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
        toml.write(toml_old.strip())


def set_python_version(version: int, sub: int):
    version_full = f"3.{version}.{sub}"
    print("Using python version", version_full)
    subprocess.check_output(["uv", "python", "pin", version_full])
    return version_full


def list_tests():
    out = os.listdir()
    files = []
    for name in out:
        if ".py" in name:
            if name in ["util.py", ".python-version"]:
                continue
            print(name)
            files.append(name)
    print("\n")
    return files


def loop_files(files: list[str], result: dict[str, str]):
    print("Running tests")
    for name in files:
        print("Running", name)
        try:
            if os.name == "posix":
                out = subprocess.check_output(["uv", "run", name])
            elif os.name == "nt":
                out = subprocess.check_output(["uv", "run", name], shell=True)
            out = out.decode("utf-8")
            print(out)
            result[name] = "Pass"
            print("Done\n")
        except subprocess.CalledProcessError as exception:
            print("Exception:", exception)
            print("Return code:", exception.returncode)
            print("Done\n")
            result[name] = "Fail"


def loop_versions(files: list[str]):
    versions = []
    results = {}

    # for version in range(10, 15):
    #     for sub in range(0, 20):

    for version in [13]:
        for sub in [7]:
            version_full = set_python_version(version, sub)
            try:
                if os.name == "posix":
                    subprocess.check_output(["uv", "sync", "--reinstall"])
                elif os.name == "nt":
                    subprocess.check_output(
                        ["uv", "sync", "--reinstall"], shell=True)
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


def assert_results(results: dict):
    for version, result in results.items():
        for name, outcome in result.items():
            if outcome == "Fail":
                raise AssertionError(f"(Python {version}) {name} failed.")


def print_results_xml(results: dict):
    with open("test-results.xml", "w") as file:
        file.write("<testsuites>\n")
        for version, result in results.items():
            tests = 0
            failures = 0
            for outcome in result.values():
                tests += 1
                if outcome == "Fail":
                    failures += 1

            file.write(
                f"<testsuite name=\"Python {version}\" "
                f"tests=\"{tests}\" failures=\"{failures}\">\n"
            )

            for name, outcome in result.items():
                if outcome == "Pass":
                    file.write(
                        f"<testcase classname=\"Python {version}\" "
                        f"name=\"{name}\"/>\n"
                    )
                else:
                    file.write(
                        f"<testcase classname=\"Python {version}\" "
                        f"name=\"{name}\">\n"
                    )
                    file.write("<failure>\n")
                    file.write("Returned error")
                    file.write("</failure>\n")
                    file.write("</testcase>\n")

            file.write("</testsuite>\n")
        file.write("</testsuites>")


def main():
    toml_old = relax_toml()

    os.chdir("tests")

    files = list_tests()

    results = loop_versions(files)

    os.chdir("..")

    print_results(results)

    set_python_version(13, 7)
    revert_toml(toml_old)

    # assert_results(results)
    print_results_xml(results)

    return results


if __name__ == "__main__":
    from pogger import Pogger as Logger
    with Logger("superspinsim-test") as logger:
        wrap = logger.record(("results"), (None))(main)
        wrap()
