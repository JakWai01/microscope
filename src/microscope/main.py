import click
import matplotlib.pyplot as plt
from qiskit import warnings

from commands.baseline import qiskit_baseline
from commands.hamiltonians import hamiltonians
from commands.microbench import microbench, microbench_new
from commands.slide import slide

import time


@click.command()
@click.argument("command", nargs=1)
@click.argument("files", nargs=-1, type=click.Path(exists=True))
@click.option("--show", type=bool, help="True if circuits should be shown")
def main(command: str, files: tuple[str, ...], show: bool):
    # Ignore deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    match command:
        case "hamiltonians":
            hamiltonians(show)
        case "microbench":
            t0 = time.perf_counter() 
            # microbench(files, show)
            microbench_new(files)
            t1 = time.perf_counter()
            print(f"Took {t1 - t0:.2f}s")
        case "slide":
            slide()
        case "baseline":
            t = time.process_time()
            qiskit_baseline(files)
            elapsed_time = time.process_time() - t
            print(f"Execution took: {elapsed_time} secs")
        case _:
            print(
                "Invalid command. Choose one out of [hamiltonians, microbench, slide, qiskit_baseline]"
            )

    plt.show()


if __name__ == "__main__":
    main()
