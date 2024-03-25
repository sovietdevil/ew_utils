import numpy as np
import scipy
from matplotlib import pyplot as plt
import rasterio
from abtem import PlaneWave
from abtem.core.energy import energy2wavelength
from torchvision.transforms import ToTensor
import hyperspy.api as hs
import atomap.api as am
import atomap.initial_position_finding as ipf

def propagation(waves, distance, sampling, energy):
    wavelength = energy2wavelength(energy)
    waves = np.array(waves)
    m, n = waves.shape
    kx = np.fft.fftfreq(m, sampling)
    ky = np.fft.fftfreq(n, sampling)
    Kx, Ky = np.meshgrid(kx, ky)
    k2 = Kx ** 2 + Ky ** 2
    kernel = np.exp(- 1.j * k2 * np.pi * wavelength * distance)
    waves = scipy.fft.ifft2(scipy.fft.fft2(waves)*kernel)
    return waves

def select_freq_range(exitwave, gmin, gmax, sampling):
    exitwave = np.array(exitwave)
    m, n = exitwave.shape
    ft_exitwave = scipy.fft.fft2(exitwave)
    freq_gx = np.fft.fftfreq(m, sampling)
    freq_gy = np.fft.fftfreq(n, sampling)
    gx, gy = np.meshgrid(freq_gx, freq_gy)
    g2 = gx ** 2 + gy ** 2
    ft_exitwave[g2 < gmin ** 2] = 0
    ft_exitwave[g2 > gmax ** 2] = 0
    ew = scipy.fft.ifft2(ft_exitwave)
    return ew

def real_space_filter(wavefunction, sampling, center=None, coeff=1, sigma=1):
    wavefunction = np.array(wavefunction)
    m, n = wavefunction.shape
    a = m*sampling/2
    if center is None:
        center = (m//2, n//2)
    x = np.linspace((1-center[0])*sampling, (m-center[0])*sampling, m)
    y = np.linspace((1-center[1])*sampling, (n-center[1])*sampling, n)
    x, y = np.meshgrid(x, y)
    r2 = x**2 + y**2
    mask = coeff * np.exp(-r2/(2*sigma))
    wave_filtered = wavefunction * mask
    wave_remained = wavefunction * (1-mask)
    return wave_filtered, wave_remained    

def Fourier_space_filter(wavefunction, sampling, coeff=1, sigma=1):
    wavefunction = np.array(wavefunction)
    m, n = wavefunction.shape
    kx = np.fft.fftfreq(m, sampling)
    ky = np.fft.fftfreq(n, sampling)
    interval = 1/(sampling*m)
    Kx, Ky = np.meshgrid(kx, ky)
    k2 = Kx ** 2 + Ky ** 2
    mask = coeff * np.exp(-k2/(2*sigma))
    wavefunc_filtered = scipy.fft.ifft2(scipy.fft.fft2(wavefunction)*mask)
    wavefunc_remained = scipy.fft.ifft2(scipy.fft.fft2(wavefunction)*(coeff-mask))
    return wavefunc_filtered, wavefunc_remained

def Average_background(wavefunction, sampling):
    wavefunction = np.array(wavefunction)
    m, n = wavefunction.shape
    wave_FT = scipy.fft.fft2(wavefunction)
    wave_FT[0,0] = (wave_FT[1,0]+wave_FT[-1,0]+wave_FT[0,1]+wave_FT[0,-1])/4
    return scipy.fft.ifft2(wave_FT)  

class ExitWave:
    def __init__(self, wavefunction, sampling, energy, peak_sites=None, plots=None, params=None):
        self.wavefunction = wavefunction
        self.sampling = sampling
        self.energy = energy
        self.amplitude = np.abs(wavefunction)
        self.phase = np.angle(wavefunction)
        self.peak_sites = peak_sites
        self.plots = plots
        self.params = params

    def show_amp(self, title="Amplitude", bar_loc="right", title_loc='center'):
        '''
        Show the amplitude of the wavefunction
        '''
        plt.imshow(self.amplitude)
        plt.colorbar(location=bar_loc)
        plt.title(title, loc=title_loc)
        plt.axis("off")

    def show_pha(self, title="Phase", bar_loc="right", title_loc='center'):
        '''
        Show the phase of the wave function
        '''
        plt.imshow(self.phase)
        plt.colorbar(location=bar_loc)
        plt.title(title, loc=title_loc)
        plt.axis("off")

    def show_fft(self, gmax=2, vmax=5e3, title="Fourier transform", bar_loc="right", title_loc='center'):
        ew_FT = np.fft.fftshift(scipy.fft.fft2(self.wavefunction))
        m, n = self.wavefunction.shape
        sampling = self.sampling
        kx = np.fft.fftfreq(m, sampling)
        ky = np.fft.fftfreq(n, sampling)
        interval = 1/(sampling*m)
        Kx, Ky = np.meshgrid(kx, ky)
        k2 = Kx ** 2 + Ky ** 2
        index =  int(gmax/interval)
        FT_sel = ew_FT[m//2-index:m//2+index, n//2-index:n//2+index]
        plt.imshow(np.abs(FT_sel), vmin=0, vmax=vmax)
        plt.colorbar(location=bar_loc)
        plt.title(title, loc=title_loc)
        plt.axis("off")

    def sel_range(self, index):
        upper, lower, left, right = index
        wavefunc_sel = self.wavefunction[upper:lower, left:right]
        return ExitWave(wavefunc_sel,
                        sampling=self.sampling,
                        energy=self.energy)

    def propagate(self, distance):
        waves = self.wavefunction
        sampling = self.sampling
        energy = self.energy
        wavefunction = propagation(waves, distance, sampling, energy)
        return ExitWave(wavefunction, sampling, energy)

    def apply_background_filter(self, coeff=1, sigma=1):
        background, wavefunction = Fourier_space_filter(self.wavefunction,
                                                        self.sampling,
                                                        coeff=coeff,
                                                        sigma=sigma)
        self.wavefunction = wavefunction
        self.amplitude = np.abs(wavefunction)
        self.phase = np.angle(wavefunction)

    def isolate_column(self, positions=None, coeff=1, sigma=1):
        wavefunction, _ = real_space_filter(self.wavefunction, 
                                            self.sampling, 
                                            positions, 
                                            coeff=coeff,
                                            sigma=sigma)
        self.wavefunction = wavefunction
        self.amplitude = np.abs(wavefunction)
        self.phase = np.angle(wavefunction)


class Columns(ExitWave):
    def __init__(self, wavefunction, sampling, energy, pca=False, separation=5, peak_sites=None, plots=None, params=None):
        super().__init__(wavefunction, sampling, energy, peak_sites, plots, params)
        self.peaks_pha = am.get_atom_positions(hs.signals.Signal2D(np.angle(wavefunction)),
                                                pca=pca, separation=separation)
        self.peaks_amp = am.get_atom_positions(hs.signals.Signal2D(np.abs(wavefunction)),
                                                pca=pca, separation=separation)
        sublattice = am.Sublattice(self.peaks_pha, 
                                   image=hs.signals.Signal2D(np.angle(wavefunction)).data,
                                   fix_negative_values=True)
        sublattice.construct_zone_axes()
        self.lattice = np.array(sublattice.zones_axis_average_distances)

def find_sites(exitwave):
    return Columns(wavefunction=exitwave.wavefunction,
                    sampling = exitwave.sampling,
                    energy=exitwave.energy)

def from_files(fname, sampling, energy, fpath='./', reverse=False):
    '''
    Extract the exit wave function from .tif files.
    Inputs:
        fname: string type, name of the .tif file
        fpath: string type, the path where the file is located, default "./"
        reverse: boolean type, whether considering the phase as inversely reconstructed, defult False
    Outputs:
        Exitwave object, assigned with a np.Complex128 array of the exit wave function
    '''
    #Check for the path name
    if fpath[-1] == '/':
        fullname = fpath+fname
    else:
        fullname = fpath+'/'+fname
    #Extract data from the file
    with rasterio.open(fullname) as image:
        image_array = image.read()
    torch_image = ToTensor()(image_array)
    real = np.transpose(torch_image.numpy()[:,0,:])
    imag = np.transpose(torch_image.numpy()[:,1,:])
    #Some exit wave are reconstructed with inverse phase
    if reverse==True:
        wavefunction = real - 1j*imag
    else:
        wavefunction = real + 1j*imag
    return ExitWave(np.array(wavefunction), sampling, energy)

def from_simulation(crystal, sampling, energy, gmax=2):
    wave = PlaneWave(energy=energy, sampling=sampling)
    wavefunction = select_freq_range(np.array(wave.multislice(crystal).array),
                                     gmin=0,
                                     gmax=gmax,
                                     sampling=sampling)
    return ExitWave(wavefunction, sampling, energy)

def defocus_stack(exitwave, start, end, step):
    wavefunction = exitwave.wavefunction
    sampling = exitwave.sampling
    energy = exitwave.energy
    defocus = np.arange(start, end, step)
    stack = []
    for deltaf in defocus:
        stack.append(propagation(wavefunction, deltaf, sampling, energy))
    return np.array(stack)
