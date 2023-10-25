# Imports 
import numpy as np
import numpy.polynomial.polynomial as poly
import scipy as sp
import astropy as ap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Loading Projection Extracted From .fits file
projection = np.loadtxt("Part 2/ds9.dat")
pixels = projection[:,0]
intensities = np.flip(projection[:,1]) # So that the plot will be from bottom->top and left->right

plt.title("Projected Spectrum of DS9 Data")
plt.plot(pixels,intensities,color="teal")
plt.xlabel("Pixel Position (Y Axis)")
plt.ylabel("Intensity")
plt.grid()
plt.show()


# Peak Finding 
min_height = 0.1 * np.max(intensities)
initial_peaks, properties = sp.signal.find_peaks(intensities, height=min_height)
peaks = np.delete(initial_peaks,[0,1,11,12]) # Removing peaks that are not OH skylines


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
range_value = 5
right_line_points = [peaks[i]+range_value for i in range(len(peaks))]
left_line_points = [peaks[i]-range_value for i in range(len(peaks))]

# Plotting Centroid Bounds
plt.title("Centroid Boundary Range")
plt.xlabel("Pixel Position")
plt.xlim(0,len(pixels))
plt.ylim(0,np.max(intensities))
plt.ylabel("Intensity")
plt.plot(pixels,intensities,label="Intensity")
plt.plot(pixels[peaks],intensities[peaks], ls='', marker="x", color="red", label="Peaks")
plt.vlines(right_line_points,color="red",ymin=0,ymax=0.3*np.max(intensities),alpha=0.5, label="Right Boundary: +{p} pixels".format(p=range_value))
plt.vlines(left_line_points,color="green",ymin=0,ymax=0.3*np.max(intensities),alpha=0.5, label="Left Boundary: -{p} pixels".format(p=range_value))
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
plt.xlim(0, len(intensities))
plt.show()

# fitting
known_lines = [16128.608, 16194.615, 16128.608, 16317.161, 
                    16350.650, 16360.385, 16442.155, 16475.648, 
                    16502.365, 16903.679]

# Finding the fits
linear_fit = poly.polyfit(centroids,known_lines,1)
quadratic_fit = poly.polyfit(centroids,known_lines,2)
cubic_fit = poly.polyfit(centroids,known_lines,3)

# Calculating Fit for all pixels
linear_values_all = poly.polyval(pixels,linear_fit)
quad_values_all = poly.polyval(pixels,quadratic_fit)
cub_values_all = poly.polyval(pixels,cubic_fit)

# Calculating fit only for centroid points
linear_values_cen = poly.polyval(centroids,linear_fit)
quad_values_cen = poly.polyval(centroids,quadratic_fit)
cub_values_cen = poly.polyval(centroids,cubic_fit)

# # Printing out centroid -> wavelength mapping
# for i in range(len(centroids)):
#     print("Identified {i} centroid as wavelength {w} nm".format(i=centroids[i],w=linear_values_cen[i]))

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

# Calculate Residuals
linear_residuals = np.array(known_lines) - np.array(linear_values_cen)
quad_residuals = np.array(known_lines) - np.array(quad_values_cen)
cub_residuals = np.array(known_lines) - np.array(cub_values_cen)

# Plot Residuals
fig, ax = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
ax[0].scatter(centroids, linear_residuals, color="red")
ax[0].set_title("Linear Fit Residuals")
ax[0].set_ylabel("Residuals")
ax[0].grid(True)
ax[1].scatter(centroids, quad_residuals, color="teal")
ax[1].set_title("Quadratic Fit Residuals")
ax[1].set_ylabel("Residuals")
ax[1].grid(True)
ax[2].scatter(centroids, cub_residuals, color="green")
ax[2].set_title("Cubic Fit Residuals")
ax[2].set_xlabel("Pixel Number")
ax[2].set_ylabel("Residuals")
ax[2].grid(True)
plt.tight_layout()
plt.subplots_adjust(top=0.95)  
plt.show()