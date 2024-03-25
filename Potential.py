from ase import Atoms
import numpy as np
import scipy
import abtem
import ew_utils
from ew_utils import ExitWave
from abtem.core.energy import energy2wavelength, energy2sigma

def generate_periodic_column_fcc(element, a, h, n):
    '''
    Generate the ase crystal object with the column of a unit cell.
    Inputs:
        element: string type, the element introduced in the crystal
        h: float type, the height between two atoms along the column
        n: int type, the number of atoms along the column
    Outputs:
        the ase Crystal object, modeling the column
    '''
    H = h*(n-1)
    sc_pos = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
          (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0),
          (1,0.5,0.5), (0.5,1,0.5)]
    cell = Atoms(element+"9", cell=[a, a, a], scaled_positions=sc_pos)
    structure = cell * (1, 1, n)
    structure.cell[2] = np.array([0, 0, (n-1/2)*h])
    return structure

def set_dopant(structure, a, h, dopant_index, dopant_sym):
    dopant_position = np.array([[a*0.5, a*0.5, h*dopant_index]])
    dopant_list = np.where((structure.positions==dopant_position[:, None]).all(-1))[1]
    original_sym = structure.get_chemical_symbols()[0]
    structure.set_chemical_symbols([dopant_sym if j in dopant_list else original_sym for j in range(len(structure))])
    return structure

def multislice_profile(potential_array, thickness, sampling, energy, gmin=0, gmax=8):
    #initilization for the storage of peaks
    peak_record = []

    #Conduct multislice simulation
    #sampling = 0.01
    energy = 300e3
    entrance_wave = np.ones(potential_array.shape[1:]) * (1 + 1j*0)
    # Core parts:
    #wave_function = multislice(entrance_wave, potential_array, sampling, H + gap, energy)
    waves = entrance_wave
    potential = potential_array
    Energy = energy
    #regular constants
    h = 6.62607015e-34
    hbar = h/(2*np.pi)
    e = 1.602176634e-19
    me = 9.10938356e-31
    c = 299792458

    #relative constants
    wave_vector = 2*np.pi/(h*c) * np.sqrt(e**2*Energy**2+2*me*c**2*e*Energy)
    wavelength = h*c/np.sqrt(e**2*Energy**2+2*me*c**2*e*Energy)
    sigma = (me*e)/(wave_vector*hbar**2)
    wavelength = energy2wavelength(energy)
    sigma = energy2sigma(energy)

    #parameters for the multislice algorithm
    slice_number = potential.shape[0]
    #convert the wave into cupy format
    waves = np.array(waves)
    m, n = waves.shape

    kx = np.fft.fftfreq(m, sampling)
    ky = np.fft.fftfreq(n, sampling)
    Kx, Ky = np.meshgrid(kx, ky)
    k2 = Kx ** 2 + Ky ** 2
    distance = thickness / slice_number
    kernel = np.exp(- 1.j * k2 * np.pi * wavelength * distance)
    dep_sec = []

    #conduct multislice calculation
    for i in range(slice_number):
        #multiply potential
        waves = waves * np.exp(1.j * sigma * np.array(potential[i,:,:]) * distance)
        #propagation
        waves = scipy.fft.ifft2(scipy.fft.fft2(waves)*kernel)
        #waves = select_freq_range(waves, 0, 1, sampling)
        #record peaks
        waves = ExitWave.select_freq_range(np.array(waves), gmin, gmax, sampling)
        peak_record.append(waves)
        m, n = waves.shape
        line_prof = waves[m//2, :]
        dep_sec.append(line_prof)

    peak_record = np.array(peak_record)
    dep_sec = np.array(dep_sec)
    return peak_record, dep_sec

def multislice_depth(structure, thickness, sampling, energy, gmin=0, gmax=8):
    potentials = abtem.Potential(structure, sampling=sampling)
    potential_array = potentials.build().compute().array
    return multislice_profile(potential_array, thickness, sampling, energy, gmin, gmax)

def generate_average_potential(structure, sampling):
    potentials = abtem.Potential(structure, sampling=sampling)
    potential_array = potentials.build().compute().array
    slice_number = potentials.shape[0]
    potential_flat = np.mean(potential_array, axis=0)
    potential_rec = np.tile(potential_flat, (slice_number, 1, 1))
    return potential_rec

def average_potential_multislice(structure, thickness, sampling, energy, gmin=0, gmax=8):
    average_potential = generate_average_potential(structure, sampling)
    return multislice_profile(average_potential, thickness, sampling, energy, gmin, gmax)