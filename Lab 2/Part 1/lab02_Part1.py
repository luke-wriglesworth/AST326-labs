# Imports
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.constants as c
import numpy as np
import numpy.polynomial.polynomial as poly
import scipy as sp

neon_data = np.loadtxt("Part 1/Ne_calib.dat")

pixels = np.arange(len(neon_data))

# Peak Finding For 1D Neon Lamp Source
min_height = 0.02 * np.max(neon_data)
initial_peaks, properties = sp.signal.find_peaks(neon_data, height=min_height)
print("Number of found peaks in neon lamp data {p}".format(p=len(initial_peaks)))

# It is detecting a peak twice at index 22, pixel 613, so I am manually deleting it because I can't get the parameters right to not detect it twice. 
peaks = np.delete(initial_peaks,[22])

plt.rcParams["figure.figsize"] = (10,6)
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
# Takes an array of positions and intensities and returns the centroid  
def centroid(x, I):
    return np.sum(x*I)/np.sum(I)

# Determining Range Value For Centroids
range_value = 5
right_line_points = [peaks[i]+range_value for i in range(len(peaks))]
left_line_points = [peaks[i]-range_value for i in range(len(peaks))]

# Plotting Centroid Bounds
plt.title("Centroid Boundary Range")
plt.xlabel("Pixel Position")
plt.xlim(500,900)
plt.ylim(0,0.7*np.max(neon_data))
plt.ylabel("Intensity")
plt.plot(pixels,neon_data,label="Intensity")
plt.plot(pixels[peaks],neon_data[peaks], ls='', marker="x", color="red", label="Peaks")
plt.vlines(right_line_points,color="red",ymin=0,ymax=0.3*np.max(neon_data),alpha=0.5, label="Right Boundary: +{p} pixels".format(p=range_value))
plt.vlines(left_line_points,color="green",ymin=0,ymax=0.3*np.max(neon_data),alpha=0.5, label="Left Boundary: -{p} pixels".format(p=range_value))
plt.legend()
plt.show()

# Calculate Centroids
centroids = [centroid(pixels[peaks[i]-range_value:peaks[i]+range_value], neon_data[peaks[i]-range_value:peaks[i]+range_value]) for i in range(len(peaks))]

# Plotting The Centroids and Residuals
fig = plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace = 0.0)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1], sharex = ax0)
ax0.plot(pixels, neon_data, color = 'purple', label = 'Data')
ax0.plot(pixels[peaks], neon_data[peaks], color = 'coral', ls = '', marker = 'x',
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
plt.xlim(0, len(neon_data))
plt.show()

# Now to use OLS fitting to find a relationship between the pixel positions and suspected wavelengths. 

true_wavelengths = [540.056,576.441,582.015,585.249,588.189,594.483,597.553,602.00,607.433,
                         614.306,616.359,621.728,626.649,630.479,633.442,638.299,640.225,650.653,
                         653.288,655.895,667.828,671.204,692.947, 703.241,717.394,724.512,743.890,
                         747.244,748.887,753.577,754.404]

# Finding the fits
linear_fit = poly.polyfit(centroids,true_wavelengths,1)
quadratic_fit = poly.polyfit(centroids,true_wavelengths,2)
cubic_fit = poly.polyfit(centroids,true_wavelengths,3)

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
plt.scatter(centroids, true_wavelengths)
plt.plot(pixels, linear_values_all)
plt.xlabel("Pixels")
plt.ylabel("True Wavelength (nm)")
plt.grid(".-")
plt.show()

plt.title("Quadratic Fit")
plt.scatter(centroids, true_wavelengths)
plt.plot(pixels,quad_values_all)
plt.xlabel("Pixel Number")
plt.ylabel("True Wavelength (nm)")
plt.grid(".-")
plt.show()

plt.title("Cubic Fit")
plt.scatter(centroids, true_wavelengths)
plt.plot(pixels,cub_values_all)
plt.xlabel("Pixel Number")
plt.ylabel("True Wavelength (nm)")
plt.grid(".-")
plt.show()

# Calculate Residuals
linear_residuals = np.array(true_wavelengths) - np.array(linear_values_cen)
quad_residuals = np.array(true_wavelengths) - np.array(quad_values_cen)
cub_residuals = np.array(true_wavelengths) - np.array(cub_values_cen)

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

# Blackbody Calculations

bb_data = np.loadtxt("Part 1/Group_S_BB.dat")
pixels = np.arange(len(bb_data))


plt.title("Raw Data of Blackbody")
plt.xlabel("Pixel Number")
plt.ylabel("Intensity")
plt.plot(pixels,bb_data,color="teal",linestyle="--")
plt.grid(".-")
plt.show()

# Using linear fit to calculate wavelengths
plt.title("Wavelength Spectrum of Blackbody")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity")
plt.plot(linear_values_all, bb_data, color="teal",linestyle="--")
plt.grid(".-")
plt.show()

# Finding Peak Intensity Wavelength -> 628.4285223741275nm
max_wavelength = linear_values_all[np.argmax(bb_data)]
print("Wavelength of Highest Intensity is : {w}nm".format(w=max_wavelength))

# Now using Wien's displacement law to find temperature
wavelength_bb_meters = max_wavelength * 10**(-9)

print("Temperature of blackbody is: {t} Kelvin".format(t = c.Wien / wavelength_bb_meters))