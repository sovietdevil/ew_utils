import numpy as np
from matplotlib import pyplot as plt
from ew_utils import ExitWave

def Argand_plot(exitwave, start, end, steps, spot=None):
    wavefunction = exitwave.wavefunction
    rec = []
    m, n = wavefunction.shape
    if spot is None:
        spot=(m//2, n//2)
    for deltaf in np.arange(start, end, steps):
        ew_prop = ExitWave.propagation(wavefunction, deltaf,
                                       sampling=exitwave.sampling,
                                       energy=exitwave.energy)
        selected = ew_prop[spot[0], spot[1]]
        rec.append(selected)
    return np.array(rec)

def draw_Argand(exitwave, start, end, steps, spot=None, color=None, label=None):
    Argand = Argand_plot(exitwave, start, end, steps, spot)
    if not (color is None) and not (label is None):
        plt.plot(np.real(Argand), np.imag(Argand), color=color, label=label)
    elif not (label is None):
        plt.plot(np.real(Argand), np.imag(Argand), label=label)
    else:
        plt.plot(np.real(Argand), np.imag(Argand))
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.axis("square")