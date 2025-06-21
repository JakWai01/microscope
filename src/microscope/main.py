import click
import matplotlib.pyplot as plt
from qiskit import warnings

from commands.baseline import qiskit_baseline
from commands.hamiltonians import hamiltonians
from commands.microbench import single, bench
from commands.slide import slide
from commands.ocular import ocular 

import time
import yaml


@click.command()
@click.argument("command", nargs=1)
@click.option("--show", type=bool, help="True if circuits should be shown")
def main(command: str, show: bool):
    # Ignore deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    config = parse_config()

    t0 = time.perf_counter()

    match command:
        case "ocular":
            ocular(config)
        case "hamiltonians":
            hamiltonians(show)
        case "single":
            single(config, show)
        case "bench":
            bench(config, show)
        case "slide":
            slide()
        case "baseline":
            qiskit_baseline(files)
        case _:
            print(
                "Invalid command. Choose one out of [hamiltonians, microbench, slide, qiskit_baseline]"
            )

    t1 = time.perf_counter()
    print(f"Took {t1 - t0:.2f}s")
    plt.show()


def parse_config():
    # Parse yaml benchmark config file
    files = []
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    main()
