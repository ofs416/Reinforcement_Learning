import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List


def energy_cons_check(
    trajectory: np.ndarray, g: float, length: float, mass: float
) -> None:
    gp = mass * g * length * (1 + np.cos(trajectory[:, 0]))
    ke = 0.5 * mass * length * trajectory[:, 1] ** 2
    total_energy = gp + ke
    print(total_energy.max() / total_energy[0])


def plot_state(
    xaxis: np.ndarray,
    yaxis: np.ndarray,
    title: Optional[str],
    legends: Optional[List[str]],
) -> None:
    plt.figure(figsize=(12, 6))
    for i in range(yaxis.shape[1]):
        if legends:
            plt.plot(xaxis, yaxis[:, i], label=legends[i])
            plt.legend()
        else:
            plt.plot(xaxis, yaxis[:, i])
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("State")
    plt.grid(True)
    plt.show()
