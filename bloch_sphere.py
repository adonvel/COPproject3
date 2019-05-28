from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import time


def plot_bloch(psi):

    psi0 = psi[0, :].tolist()
    psi1 = psi[1, :].tolist()
    b = Bloch()
    b.point_marker = "o"
    b.point_color = "b"
    b.point_size = [0.7]
    b.view = [0, 0]
    b.add_states([state for state in psi0*basis(2, 0) + psi1*basis(2, 1)], "point")
    b.show()


def animate_bloch(psi):

    psi0 = psi[0, :].tolist()
    psi1 = psi[1, :].tolist()
    phis = [270] # np.linspace(0, 360, 30)
    i = 0
    for phi in phis:
        print(phi)
        b = Bloch()
        b.point_marker = "o"
        b.point_color = "b"
        b.point_size = [0.7]
        b.add_states([state for state in psi0 * basis(2, 0) + psi1 * basis(2, 1)], "point")
        b.view = [int(phi), 0]
        b.save(dirc="temp" + str(i))
        i += 1
