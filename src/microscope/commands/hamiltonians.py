import os
from commands.microbench import run


def hamiltonians(show):
    path = "/home/jakob/Documents/hamiltonians"
    files = os.listdir(path)

    for file in files:
        es, swaps = run(path + "/" + file, show)
        print(f"File: {file}\nExtended Set Size: {es}\nSwaps: {swaps}")
