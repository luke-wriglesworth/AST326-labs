# Imports 
import numpy as np
import numpy.polynomial.polynomial as poly
import scipy as sp
import scipy.constants as c
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Loading Projections Extracted Using DS9. Each sample has the exact same dimensions, 
# only shifted in the spatial direction.
projection = np.loadtxt("Part 2/ds9_1.dat")

# DS9 Y Pixel Ranges
ymin = 52
ymax = 217

pixels = np.arange(ymin,ymax)
intensities = projection[:,1]

emission = np.loadtxt("Part 2/FE_1.dat")
emission_pixels = np.arange(ymin,ymax)
emission_intensities = emission[:,1]

plt.title("Projected Spectrum of DS9 Data")
plt.plot(pixels,intensities,color="teal")
plt.xlabel("Pixel Position (Y Axis)")
plt.ylabel("Intensity")
plt.grid()
plt.show()

# Peak Finding 
min_height = 0.1 * np.max(intensities)
initial_peaks, properties = sp.signal.find_peaks(intensities, height=min_height)
peaks = np.delete(initial_peaks,[0,6,8,9,10,12,13]) # Removing peaks that are not OH skylines

# Peak Plotting
plt.rcParams["figure.figsize"] = (10,6)
plt.rcParams['font.size'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.title("DS9 Data With {n} Identified Peaks".format(n=len(peaks)))
plt.xlabel("Pixel Position")
plt.ylabel("Intensity")
plt.plot(pixels,intensities,label="Intensity")
plt.plot(pixels[peaks],intensities[peaks], ls='', marker="x", color="red", label="Peaks")
plt.grid(ls = '-')
plt.legend()
plt.show()

# Finding Centroids
def centroid(x, I):
    return np.sum(x*I)/np.sum(I)

# Determining Range Value For Centroids
range_value = 3
right_line_points = [peaks[i]+range_value + ymin for i in range(len(peaks))]
left_line_points = [peaks[i]-range_value + ymin for i in range(len(peaks))]
print(right_line_points)
# Plotting Centroid Bounds
plt.title("Centroid Boundary Range")
plt.xlabel("Pixel Position")
plt.xlim(np.min(pixels),np.max(pixels))
plt.ylim(0,np.max(intensities))
plt.ylabel("Intensity")
plt.plot(pixels,intensities,label="Intensity")
plt.plot(pixels[peaks],intensities[peaks], ls='', marker="x", color="red", label="Peaks")
plt.vlines(right_line_points,color="red",ymin=0,ymax=.9*np.max(intensities),alpha=0.5, label="Right Boundary: +{p} pixels".format(p=range_value))
plt.vlines(left_line_points,color="green",ymin=0,ymax=.9*np.max(intensities),alpha=0.5, label="Left Boundary: -{p} pixels".format(p=range_value))
plt.legend()
plt.show()

# Calculate Centroids
centroids = [centroid(pixels[peaks[i]-range_value:peaks[i]+range_value], intensities[peaks[i]-range_value:peaks[i]+range_value]) for i in range(len(peaks))]

# Plotting The Centroids and Residuals
fig = plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace = 0.0)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1], sharex = ax0)
ax0.plot(pixels, intensities, color = 'purple', label = 'Data')
ax0.plot(pixels[peaks], intensities[peaks], color = 'coral', ls = '', marker = 'x',
         ms = 5, label = 'Peaks')
[ax0.axvline(c, color = 'teal', alpha = 0.5, label = 'Centroids') if c == centroids[0]
 else ax0.axvline(c, color = 'teal', alpha = 0.5) for c in centroids]
ax1.plot(pixels[peaks], pixels[peaks] - centroids, ls = '', marker = '.', color = 'coral')
ax1.set_xlabel('Pixels')
ax0.set_ylabel('Intensity')
ax1.set_ylabel('Residuals')
plt.setp(ax0.get_xticklabels(), visible = False)
ax0.legend(loc = 'best', fontsize = 10)
ax0.grid(ls = '-.')
ax1.grid(ls = '-.')
plt.xlim(np.min(pixels), np.max(pixels))
plt.show()

#fitting
# known_lines = [16128.608, 16194.615, 16235.376, 16317.161, 
#                16350.650, 16442.155, 16447.616, 16502.365, 
#                16553.814]

known_lines = [16128.608, 16194.615, 16235.376, 16317.161, 
               16350.650, 16442.155, 16502.365, 
               16553.814]

# Finding the fits
linear_fit = poly.polyfit(centroids,known_lines,1)
quadratic_fit = poly.polyfit(centroids,known_lines,2)
cubic_fit = poly.polyfit(centroids,known_lines,3)

# Calculating Fit for all pixels
linear_values_all = poly.polyval(pixels,linear_fit)
quad_values_all = poly.polyval(pixels,quadratic_fit)
cub_values_all = poly.polyval(pixels,cubic_fit)

# Calculating fit only for centroid points to be used for residuals
linear_values_cen = poly.polyval(centroids,linear_fit)
quad_values_cen = poly.polyval(centroids,quadratic_fit)
cub_values_cen = poly.polyval(centroids,cubic_fit)

# Calculate Residuals
linear_residuals = np.array(known_lines) - np.array(linear_values_cen)
quad_residuals = np.array(known_lines) - np.array(quad_values_cen)
cub_residuals = np.array(known_lines) - np.array(cub_values_cen)

plt.title("Linear Fit")
plt.scatter(centroids, known_lines)
plt.plot(pixels,linear_values_all)
plt.xlabel("Pixels")
plt.ylabel("True Wavelength (Angstroms)")
plt.grid(".-")
plt.show()

plt.title("Quadratic Fit")
plt.scatter(centroids, known_lines)
plt.plot(pixels,quad_values_all)
plt.xlabel("Pixel Number")
plt.ylabel("True Wavelength (Angstroms)")
plt.grid(".-")
plt.show()

plt.title("Cubic Fit")
plt.scatter(centroids, known_lines)
plt.plot(pixels,cub_values_all)
plt.xlabel("Pixel Number")
plt.ylabel("True Wavelength (Angstroms)")
plt.grid(".-")
plt.show()

# Plot Residuals
bound = np.max(linear_residuals) + 10
fig, ax = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
ax[0].scatter(centroids, linear_residuals, color="red")
ax[0].set_title("Linear Fit")
ax[0].set_ylim(-bound,bound)
ax[0].grid(True)
ax[1].scatter(centroids, quad_residuals, color="teal")
ax[1].set_title("Quadratic Fit")
ax[1].set_ylabel("Residuals")
ax[1].set_ylim(-bound,bound)
ax[1].grid(True)
ax[2].scatter(centroids, cub_residuals, color="green")
ax[2].set_title("Cubic Fit")
ax[2].set_xlabel("Pixel Number")
ax[2].set_ylim(-bound,bound)
ax[2].grid(True)
plt.tight_layout()
plt.subplots_adjust(top=0.95)  
plt.show()

# FE2 Emission Occurs at the third to last peak by visual inspection. I will create another
# DS9 projection which includes the FE2 emission for analysis, to rigorously find the peak using 
# the centroid of it. 

plt.title("Projected Spectrum With FE2 Emission")
plt.plot(emission_pixels,emission_intensities,color="teal")
plt.xlabel("Pixel Position (Y Axis)")
plt.ylabel("Intensity")
plt.grid()
plt.show()

# Now applying wavelength solution to the new data, first I will find the centroid of the FE2 
# emission peak. 

emission_peak, properties = sp.signal.find_peaks(emission_intensities,height=0.97*np.max(emission_intensities))

# Calculate Centroid Boundaries
range_value=3
right_emission_boundary = emission_peak[0]+range_value + ymin
left_emission_boundary = emission_peak[0]-range_value + ymin

# Plotting Centroid Bounds
plt.title("Emission Centroid Boundary")
plt.xlabel("Pixel Position")
plt.xlim(ymin,ymax)
plt.ylim(0,np.max(emission_intensities))
plt.ylabel("Intensity")
plt.plot(emission_pixels,emission_intensities,label="Intensity")
plt.scatter(emission_peak[0],emission_intensities[emission_peak[0]], ls='', marker="x", color="red", label="Peaks")
plt.vlines(right_emission_boundary,color="red",ymin=0,ymax=np.max(emission_intensities),alpha=0.5, label="Right Boundary: +{p} pixels".format(p=range_value))
plt.vlines(left_emission_boundary,color="green",ymin=0,ymax=np.max(emission_intensities),alpha=0.5, label="Left Boundary: -{p} pixels".format(p=range_value))
plt.legend()
plt.show()

# Calculate Centroids
peak_centroid = [centroid(emission_pixels[emission_peak[i]-range_value:emission_peak[i]+range_value], emission_intensities[emission_peak[i]-range_value:emission_peak[i]+range_value]) for i in range(len(emission_peak))]

# Find wavelength of FE2 Emission Based On Cubic Fit
observed_wavelength = poly.polyval(peak_centroid[0],quadratic_fit)
print("Wavelength of FE2 Emission is: {w} Angstroms".format(w=observed_wavelength))

# Plotting The Peak Centroid
fig = plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace = 0.0)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1], sharex = ax0)
ax0.plot(emission_pixels, emission_intensities, color = 'purple', label = 'Data')
ax0.axvline(peak_centroid[0], color = 'teal', label = 'Peak Emission Centroid Pixel Number: {p}'.format(p=peak_centroid[0]))
ax0.plot(emission_pixels[emission_peak], emission_intensities[emission_peak], color = 'coral', ls = '', marker = 'x',
         ms = 5, label = 'Peaks')
ax1.plot(emission_pixels[emission_peak], emission_pixels[emission_peak] - peak_centroid, ls = '', marker = '.', color = 'coral')
ax1.set_xlabel('Pixels')
ax0.set_ylabel('Intensity')
ax1.set_ylabel('Residuals')
plt.setp(ax0.get_xticklabels(), visible = False)
ax0.legend(loc = 'best', fontsize = 10)
ax0.grid(ls = '-.')
ax1.grid(ls = '-.')
plt.xlim(ymin, ymax)
plt.ylim(0,np.max(emission_intensities)+10)
plt.show()

# Calculating Velocity of Emission Using Doppler Shift Formula
intrensic_wavelength = 1.6439981 # microns
intrensic_wavelength_angstroms = intrensic_wavelength*1e4 # angstroms
print(intrensic_wavelength_angstroms)
print(observed_wavelength)
velocity = c.c * (observed_wavelength-intrensic_wavelength_angstroms)/intrensic_wavelength_angstroms
print("Velocity of FE2 Emission is {v} km/s".format(v=velocity*1e-3))