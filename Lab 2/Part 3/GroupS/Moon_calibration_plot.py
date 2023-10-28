# Imports 
import numpy as np
import numpy.polynomial.polynomial as poly
from numpy.polynomial.polynomial import Polynomial
import scipy as sp
import scipy.constants as c
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Loading Neon Sample
projection = np.loadtxt("Part 3/GroupS/processed_fits/neon_spectrum.dat")

# DS9 Y Pixel Ranges
xmin = 0
xmax = 1392

f=11 # font size

pixels = np.arange(xmin,xmax)
moon_pixels = np.arange(xmin,xmax)
intensities = projection[:,1]

# moon Spectrum
moon = np.loadtxt("Part 3/GroupS/processed_fits/moon_spectrum.dat")
moon_intensities = moon[:,1] + abs(np.min(moon[:,1]))

plt.rcParams["figure.figsize"] = (10,6)
plt.rcParams['font.size'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

plt.title("Neon Spectrum Used For Calibration")
plt.plot(pixels,intensities,color="teal", label="Intensity")
plt.xlim(1,xmax+10)
plt.xlabel("Pixel Position (X Axis)")
plt.ylabel("Intensity")
plt.grid()
plt.legend(loc = 'best', fontsize = f)
plt.show()

# Peak Finding 
min_height = 0.07 * np.max(intensities)
initial_peaks, properties = sp.signal.find_peaks(intensities, height=min_height)
peaks = np.delete(initial_peaks,[2,3,4,11]) # Removing Peaks
peaks = np.array(peaks)

# Peak Plotting
plt.rcParams["figure.figsize"] = (10,6)
plt.rcParams['font.size'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.xlim(1,xmax+10)
plt.title("Neon Lamp Spectrum With {n} Identified Peaks".format(n=len(peaks)))
plt.xlabel("Pixel Position (X Axis)")
plt.ylabel("Intensity")
plt.plot(pixels,intensities,label="Intensity")
plt.plot(pixels[peaks],intensities[peaks], ls='', marker="x", color="red", label="Peaks")
plt.grid(ls = '-')
plt.legend(loc = 'best', fontsize = f)
plt.show()

# Finding Centroids
def centroid(x, I):
    return np.sum(x*I)/np.sum(I)

# Determining Range Value For Centroids
range_value = 6
right_line_points = [pixels[peaks[i]]+range_value for i in range(len(peaks))]
left_line_points = [pixels[peaks[i]]-range_value for i in range(len(peaks))]
# Plotting Centroid Bounds
plt.title("Centroid Boundary Ranges")
plt.xlabel("Pixel Position (X Axis)")
plt.xlim(1100,xmax)
plt.ylim(0,np.max(intensities)+10)
plt.ylabel("Intensity")
plt.plot(pixels,intensities,label="Intensity")
plt.plot(pixels[peaks],intensities[peaks], ls='', marker="x", color="red", label="Peaks")
plt.vlines(right_line_points,color="red",ymin=0,ymax=.9*np.max(intensities),alpha=0.3, linestyle="--", label="Right Boundary: +{p} pixels".format(p=range_value))
plt.vlines(left_line_points,color="green",ymin=0,ymax=.9*np.max(intensities),alpha=0.3, linestyle="--", label="Left Boundary: -{p} pixels".format(p=range_value))
plt.legend(loc = 'best', fontsize = f)
plt.show()

# Calculate Centroids
centroids = [centroid(pixels[peaks[i]-range_value:peaks[i]+range_value], intensities[peaks[i]-range_value:peaks[i]+range_value]) for i in range(len(peaks))]
#centroids = np.round(centroids_flt) # Rounding to nearest pixel

# Plotting The Centroids and Residuals
fig = plt.figure(figsize=(10,6))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace = 0.1)
ax0 = plt.subplot(gs[0])
ax0.set_xlim(1100,np.max(pixels))
ax0.set_title("Peak Centroids")
ax1 = plt.subplot(gs[1], sharex = ax0)
ax1.set_ylim(-0.5,0.5)
ax0.plot(pixels, intensities, color = 'purple', label = 'Intensity')
ax0.plot(pixels[peaks], intensities[peaks], color = 'coral', ls = '', marker = 'x',
         ms = 5, label = 'Peaks')
[ax0.axvline(c, color = 'teal', alpha = 0.7, label = 'Centroids') if c == centroids[0]
 else ax0.axvline(c, color = 'teal', alpha = 0.7) for c in centroids]
ax1.plot(pixels[peaks], pixels[peaks] - centroids, ls = '', marker = '.', color = 'coral')
ax1.set_xlabel('Pixel Position (X Axis)')
ax0.set_ylabel('Intensity')
ax1.set_ylabel('Residual')
plt.setp(ax0.get_xticklabels(), visible = False)
ax0.legend(loc = 'best', fontsize = f)
ax0.grid(ls = '-.')
ax1.grid(ls = '-.')
plt.show()



#fitting
known_lines = [420.067,585.249,692.947, 
               703.241, 724.517,743.89,
               750.387,763.511]


# Finding the fits 
linear_fit, linear_stats = poly.polyfit(centroids,known_lines,1,full=True)
lin = Polynomial(linear_fit)
quadratic_fit, quad_stats = poly.polyfit(centroids,known_lines,2,full=True)
quad = Polynomial(quadratic_fit)
cubic_fit, cub_stats = poly.polyfit(centroids,known_lines,3,full=True)
cub=Polynomial(cubic_fit)

# Calculating Fit for all pixels
linear_values_all = lin(pixels)
quad_values_all = quad(pixels)
cub_values_all = cub(pixels)

# Calculating fit only for centroid point
linear_values_cen = poly.polyval(centroids,linear_fit)
quad_values_cen = poly.polyval(centroids,quadratic_fit)
cub_values_cen = poly.polyval(centroids,cubic_fit)
values=[linear_values_all,quad_values_all,cub_values_all]

# Calculate Residuals
linear_residuals = np.array(known_lines) - np.array(linear_values_cen)
quad_residuals = np.array(known_lines) - np.array(quad_values_cen)
cub_residuals = np.array(known_lines) - np.array(cub_values_cen)
residuals=[linear_residuals,quad_residuals,cub_residuals]

# R^2 Values
print("R^2 Values")
print("----------")
r_array = []
for val in range(0,len(values)):
    total_square_sum = np.sum((values[val] - np.mean(values[val]))**2)  # total sum of squares
    res_square_sum = np.sum(residuals[val]**2)  # residual sum of squares
    r_squared = 1 - (res_square_sum / total_square_sum)  # R^2 value
    print("R^2 = {r}".format(r=r_squared))
    r_array.append(r_squared)

plt.suptitle("Linear Fit")
plt.title("$R^2 = ${r}".format(r=r_array[0]),fontsize=f+3)
plt.scatter(centroids, known_lines, label="Centroid Wavelength Mapping", color="red")
plt.plot(pixels,lin(pixels))
plt.xlabel("Pixel Position (X Axis)")
plt.ylabel("True Wavelength (nm)")
plt.grid(".-")
plt.legend(loc = 'best', fontsize = f)
plt.show()

plt.suptitle("Quadratic Fit")
plt.title("$R^2 = ${r}".format(r=r_array[1]),fontsize=f+3)
plt.scatter(centroids, known_lines, label="Centroid Wavelength Mapping", color="red")
plt.plot(pixels,quad(pixels))
plt.xlabel("Pixel Position (X Axis)")
plt.ylabel("True Wavelength (nm)")
plt.grid(".-")
plt.legend(loc = 'best', fontsize = f)
plt.show()

plt.suptitle("Cubic Fit")
plt.title("$R^2 = ${r}".format(r=r_array[2]),fontsize=f+3)
plt.scatter(centroids, known_lines, label="Centroid Wavelength Mapping", color="red")
plt.plot(pixels,cub(pixels))
plt.xlabel("Pixel Position (X Axis)")
plt.ylabel("True Wavelength (nm)")
plt.grid(".-")
plt.legend(loc = 'best', fontsize = f)
plt.show()

plt.title("Neon Lamp Wavelengths")
plt.plot(lin(pixels),intensities,color="teal", label="Intensity")
plt.xlim(lin(1),lin(xmax+10))
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity")
plt.grid()
plt.legend(loc = 'best', fontsize = f)
plt.show()

# Plot Residuals
bound = np.max(linear_residuals) + 10
fig, ax = plt.subplots(3, 1, figsize=(6, 7), sharex=True)
ax[0].scatter(centroids, residuals[0], color="red",label="Linear")
ax[0].set_title("Residuals")
ax[0].set_ylim(-bound,bound)
ax[0].set_xlim(300,xmax)
ax[0].grid(True)
ax[0].legend(loc = 'best', fontsize = f)
ax[1].scatter(centroids, residuals[1], color="teal", label="Quadratic")
ax[1].set_ylim(-bound,bound)
ax[1].set_xlim(300,xmax)
ax[1].legend(loc = 'best', fontsize = f)
ax[1].grid(True)
ax[2].scatter(centroids, residuals[2], color="green", label="Cubic")
ax[2].set_xlabel("Pixel Number")
ax[2].set_xlim(300,xmax)
ax[2].set_ylim(-bound,bound)
ax[2].legend(loc = 'best', fontsize = f)
ax[2].grid(True)
plt.tight_layout()
plt.subplots_adjust(top=0.95)  
plt.show()

# Plotting moon
plt.figure(figsize=(12,6))
plt.title("Moon Spectrum")
plt.plot(lin(pixels),moon_intensities,color="teal", label="Intensity")
plt.xlim(300,lin(xmax))
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity")
plt.grid()
plt.legend(loc = 'best', fontsize = f)
plt.show()