def main():
    import sys

    print("Python version", sys.version)

    raise Exception("Testing error detection")


if __name__ == "__main__":
    main()
