import click
import matplotlib.pyplot as plt
from qiskit import warnings

from commands.ocular import ocular

import time
import yaml


@click.command()
@click.argument("command", nargs=1)
def main(command: str):
    # Ignore deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    config = parse_config()

    t0 = time.perf_counter()

    match command:
        case "ocular":
            ocular(config)
        case _:
            print(
                "Invalid command. Choose one out of [hamiltonians, microbench, slide, qiskit_baseline]"
            )

    t1 = time.perf_counter()
    print(f"Took {t1 - t0:.2f}s")
    plt.show()


def parse_config():
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    main()
