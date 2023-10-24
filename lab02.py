# Imports
import numpy as np
import astropy as ap
import scipy as sp
import matplotlib.pyplot as plt

neon_data = np.loadtxt("Ne_calib.dat")
pixels = np.linspace(0,len(neon_data),len(neon_data))

# Peak Finding For 1D Neon Lamp Source
min_height = 0.03 * np.max(neon_data)
peaks, properties = sp.signal.find_peaks(neon_data, height=min_height)
print("Number of found peaks in neon lamp data {p}".format(p=len(peaks)))

plt.rcParams["figure.figsize"] = (15,5)
plt.rcParams['font.size'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

plt.title("Neon Lamp Calibration Data With {n} Identified Peaks".format(n=len(peaks)))
plt.xlabel("Pixel Position")
plt.ylabel("Intensity")
plt.plot(pixels,neon_data,label="Intensity")
plt.plot(pixels[peaks],neon_data[peaks], ls='', marker="x", color="red", label="Peaks")
plt.grid(ls = '-')
plt.legend()
plt.show()

# Now To Determine The Centroids
# Takes an array of positions and intensities and returns  
def centroid(x, I):
    return np.sum(x,I)/np.sum(I)

