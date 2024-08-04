#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import random
import math
from multiprocessing import Pool
import time as timer

tic = timer.time()

num_particles = 100
a = 0.5 * 10 ** (-7)
spring_constant = 1e-5
BOLTZMANN_CONST = 1.380649 * 10 ** (-23)
temperature = 310

num_moves = 10000000
gammas = [10000, 100000, 1000000]
diameter0s = [1e-6]  # , 5e-6, 10e-6]
diameter0 = 1e-6
step_sizes = [0.05 * a, 0.1 * a, 0.15 * a, 0.2 * a, 0.25 * a]
p_scalings = [2, 4]
d0 = 0.5 * a
h0 = 5e-6


def polyarea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def simulate(params):
    step_size, p_scaling, gamma = params
    area0 = (np.pi * diameter0**2) / 4
    particles = [{"x": 0, "y": 0, "k": spring_constant, "mtf": 0} for _ in range(num_particles)]

    time = []
    area_of_shape = []
    relative_area = []
    E_combined_total = []

    count = 1
    for t in range(1, num_moves + 1):
        c = random.randint(6, num_particles - 5)

        x_old = [particle["x"] for particle in particles]
        y_old = [particle["y"] for particle in particles]

        d1_old = distance.euclidean((x_old[c], y_old[c]), (x_old[c - 1], y_old[c - 1]))
        d2_old = distance.euclidean((x_old[c], y_old[c]), (x_old[c + 1], y_old[c + 1]))

        EN1_old = (particles[c]["k"] * (d1_old - d0) ** 2) / 2
        EP1_old = (particles[c]["k"] * (d2_old - d0) ** 2) / 2

        Espring_old = EN1_old + EP1_old

        Emt_old = particles[c]["mtf"] * (y_old[c] - h0) ** 2

        area_old = polyarea(np.array(x_old), np.array(y_old))
        Earea_old = gamma * ((area_old - area0) ** 2)

        E_total_old = Espring_old + Earea_old - Emt_old

        # Make a potential move of a particle
        x_new, y_new = x_old.copy(), y_old.copy()
        x_new[c] += np.random.uniform(-step_size, step_size)
        y_new[c] += np.random.uniform(-step_size, step_size)

        # Calculate the new energy
        d1 = distance.euclidean((x_new[c], y_new[c]), (x_new[c - 1], y_new[c - 1]))
        d2 = distance.euclidean((x_new[c], y_new[c]), (x_new[c + 1], y_new[c + 1]))

        EN1 = (particles[c]["k"] * (d1 - d0) ** 2) / 2
        EP1 = (particles[c]["k"] * (d2 - d0) ** 2) / 2

        Espring = EN1 + EP1

        Emt = particles[c]["mtf"] * (y_new[c] - h0) ** 2

        area = polyarea(np.array(x_new), np.array(y_new))
        Earea = gamma * ((area - area0) ** 2)

        E_total_new = Espring + Earea - Emt

        # Calculate the probability based on the difference in energy
        deltaE = E_total_new - E_total_old

        if deltaE < 0 and y_new[c] >= 0:
            particles[c]["x"] = x_new[c]
            particles[c]["y"] = y_new[c]
        else:
            P = np.exp((-deltaE / (BOLTZMANN_CONST * temperature)))
            r = np.random.rand()
            if P > p_scaling * r and y_new[c] >= 0:
                particles[c]["x"] = x_new[c]
                particles[c]["y"] = y_new[c]

        xpos2 = [particle["x"] for particle in particles]
        ypos2 = [particle["y"] for particle in particles]

        if t % 100000 == 0:
            time.append(t)
            area_of_shape.append(area_old)
            relative_area.append(area_old / area0)

            E_spring_tot = sum(0.5 * particles[i]["k"] * ((x_old[i] - x_old[i + 1]) ** 2 + (y_old[i] - y_old[i + 1]) ** 2) for i in range(6, num_particles - 5))

            E_combined_total.append(E_spring_tot + Earea_old - Emt_old)

            count += 1

    plt.figure(figsize=(4, 6))

    plt.subplot(3, 1, 1)
    plt.plot(time, area_of_shape, "*")
    plt.title("Area growth")
    plt.xlabel("Time")
    plt.ylabel("Area")

    plt.subplot(3, 1, 2)
    plt.plot(xpos2, ypos2, linewidth=3)
    plt.title("Bleb shape")
    plt.xlabel("X_pos")
    plt.ylabel("Y_pos")

    plt.subplot(3, 1, 3)
    plt.plot(E_combined_total, "*r")
    plt.title("Energy change")
    plt.xlabel("snapshot")
    plt.ylabel("energy")

    plt.tight_layout()

    plt.savefig(f"bleb_plots/combined_plots_step_size_{step_size}_pscaling{p_scaling}_gamma{gamma}.png")

    plt.show()


if __name__ == "__main__":
    tic = timer.time()
    param_sets = [(step_size, p_scaling, gamma) for step_size in step_sizes for p_scaling in p_scalings for gamma in gammas]

    with Pool(processes=4) as pool:
        pool.map(simulate, param_sets)

    tok = timer.time()
    print(f"It took {round(tok - tic, 2)} secs")
