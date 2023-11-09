# Imports 
import numpy as np
from numpy import polyfit
from numpy import poly1d
from numpy import polyder
import scipy as sp
import scipy.constants as c
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Loading Projections Extracted Using DS9. Each sample has the exact same dimensions, 
# only shifted in the spatial direction.
projection = np.loadtxt("Lab 2/Part 2/ds9_2.dat")

known_lines = [16128.608,16194.615,16235.376,16317.161,16350.650
               ,16442.155,16502.365,16553.814]

# DS9 Y Pixel Ranges
ymin = 52
ymax = 217

f=11 # font size

pixels = np.arange(ymin,ymax)
emission_pixels = np.arange(ymin,ymax)
intensities = projection[:,1]

emission = np.loadtxt("Lab 2/Part 2/FE_2.dat")
emission_intensities = emission[:,1]

plt.rcParams["figure.figsize"] = (10,6)
plt.rcParams['font.size'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

plt.title("Spectrum With Isolated OH Emission Lines")
plt.plot(pixels,intensities,color="teal", label="Intensity")
plt.xlabel("Pixel Position (Y Axis)")
plt.ylabel("Intensity")
plt.grid()
plt.legend(loc = 'best', fontsize = f)
plt.savefig("Lab 2/plots/OH_pixel_intensity.pdf")
plt.show()

# Peak Finding 
min_height = 0.1 * np.max(intensities)
initial_peaks, properties = sp.signal.find_peaks(intensities, height=min_height)
peaks = np.delete(initial_peaks,[0,6,8,9,11,13]) # Removing peaks that are not OH skylines
peaks = np.array(peaks)

# Peak Plotting
plt.rcParams["figure.figsize"] = (10,6)
plt.rcParams['font.size'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

plt.title("{n} Identified Peaks".format(n=len(peaks)))
plt.xlabel("Pixel Position (Y Axis)")
plt.ylabel("Relative")
plt.plot(pixels,intensities,label="Intensity")
plt.plot(pixels[peaks],intensities[peaks], ls='', marker="x", color="red", label="Identified Peak")
for i, peak in enumerate(peaks):
    #if i % 2 == 0:
    plt.annotate(str(known_lines[i])+" (A)", (pixels[peak], intensities[peak]), 
                    xytext=(20,5), textcoords='offset points', 
                    arrowprops=dict(facecolor='black', arrowstyle='->',connectionstyle="angle,angleA=0,angleB=90,rad=10"),
                    label="Wavelengths (nm)", fontsize = f-3)
plt.grid(ls = '-')
plt.legend(loc = 'best', fontsize = f)
plt.savefig("Lab 2/plots/OH_Peaks.pdf")
plt.show()

# Finding Centroids
def centroid(x, I):
    return np.sum(x*I)/np.sum(I)

# Determining Range Value For Centroids
range_value = 3
right_line_points = [pixels[peaks[i]]+range_value for i in range(len(peaks))]
left_line_points = [pixels[peaks[i]]-range_value for i in range(len(peaks))]

# Plotting Centroid Bounds
plt.title("Centroid Boundary Ranges")
plt.xlabel("Pixel Position (Y Axis)")
plt.ylim(0,np.max(intensities)+10)
plt.ylabel("Intensity")
plt.plot(pixels,intensities,label="Intensity")
plt.plot(pixels[peaks],intensities[peaks], ls='', marker="x", color="red", label="Peaks")
plt.vlines(right_line_points,color="red",ymin=0,ymax=.9*np.max(intensities),alpha=0.3, linestyle="--", label="Right Boundary: +{p} pixels".format(p=range_value))
plt.vlines(left_line_points,color="green",ymin=0,ymax=.9*np.max(intensities),alpha=0.3, linestyle="--", label="Left Boundary: -{p} pixels".format(p=range_value))
plt.legend(loc = 'best', fontsize = f)
plt.grid(".-")
plt.savefig("Lab 2/plots/OH_Centroid_Boundaries.pdf")
plt.show()

# Calculate Centroids
centroids = [centroid(pixels[peaks[i]-range_value:peaks[i]+range_value], intensities[peaks[i]-range_value:peaks[i]+range_value]) for i in range(len(peaks))]
centroid_residuals = np.array(centroids) - np.array(pixels[peaks])

# root mean square error for centroids
rmse_centroid = np.sqrt(np.mean(centroid_residuals**2))
print("Centroid RMS Error: {r}".format(r=rmse_centroid))

# Plotting The Centroids and Residuals
fig = plt.figure(figsize=(10,6))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace = 0.1)
ax0 = plt.subplot(gs[0])
ax0.set_title("Centroid Pixel Positions")
ax1 = plt.subplot(gs[1], sharex = ax0)
ax1.set_ylim(-0.5,0.5)
ax0.plot(pixels, intensities, color = 'purple', label = 'Intensity')
ax0.plot(pixels[peaks], intensities[peaks], color = 'coral', ls = '', marker = 'x',
         ms = 5, label = 'Peaks')
[ax0.axvline(c, color = 'teal', alpha = 0.7, label = 'Centroids') if c == centroids[0]
 else ax0.axvline(c, color = 'teal', alpha = 0.7) for c in centroids]
ax1.plot(pixels[peaks], pixels[peaks] - centroids, ls = '', marker = '.', color = 'coral')
ax1.set_xlabel('Pixel Position (Y Axis)')
ax0.set_ylabel('Intensity')
ax1.set_ylabel('Residual')
plt.setp(ax0.get_xticklabels(), visible = False)
ax0.legend(loc = 'best', fontsize = f)
ax0.grid(ls = '-.')
ax1.grid(ls = '-.')
plt.savefig("Lab 2/plots/OH_Centroids_Residuals.pdf")
plt.show()

#fitting
known_lines = [16128.608,16194.615,16235.376,16317.161,16350.650
               ,16442.155,16502.365,16553.814]


# Finding the fits 
linear_fit, linear_cov = polyfit(centroids,known_lines,1,full=False, cov=True)
lin = poly1d(linear_fit)
quadratic_fit, quad_cov = polyfit(centroids,known_lines,2,full=False, cov=True)
quad = poly1d(quadratic_fit)
cubic_fit, cub_cov = polyfit(centroids,known_lines,3,full=False, cov=True)
cub=poly1d(cubic_fit)

# Calculating Fit for all pixels
linear_values_all = lin(pixels)
quad_values_all = quad(pixels)
cub_values_all = cub(pixels)

# Calculating fit only for centroid point
linear_values_cen = lin(centroids)
quad_values_cen = quad(centroids)
cub_values_cen = cub(centroids)

values=[linear_values_all,quad_values_all,cub_values_all]

# Calculate Residuals
linear_residuals = np.array(known_lines) - np.array(linear_values_cen)
quad_residuals = np.array(known_lines) - np.array(quad_values_cen)
cub_residuals = np.array(known_lines) - np.array(cub_values_cen)
residuals=[linear_residuals,quad_residuals,cub_residuals]

# root mean square for residuals
rmse = []
for res in residuals:
    rmse.append(np.sqrt(np.mean(res**2)))

# Calculate Uncertainty of Fit For Coefficients
linear_uncertainty = np.sqrt(np.diag(linear_cov))
quad_uncertainty = np.sqrt(np.diag(quad_cov))
cub_uncertainty = np.sqrt(np.diag(cub_cov))
uncertainty_coefficients = [linear_uncertainty,quad_uncertainty,cub_uncertainty]
print("Polynomial Coefficient Uncertainties")
print("----------")
for u in uncertainty_coefficients:
    print(u)

linear_coef_names = ['m', 'b']
quad_coef_names = ['a', 'b', 'c']
cub_coef_names = ['a', 'b', 'c', 'd']
coef_names = [linear_coef_names, quad_coef_names, cub_coef_names]
fit_names = ['Linear', 'Quadratic', 'Cubic']
fig, ax = plt.subplots()
current_x = 0
for i, uncertainty in enumerate(uncertainty_coefficients):
    x_values = [f"{fit_names[i]} {coef_names[i][j]}" for j in range(len(uncertainty))]
    ax.bar(x_values, uncertainty, width=0.5, align="center", label="{f}".format(f=fit_names[i]))
    current_x += len(uncertainty)
    if i < len(uncertainty_coefficients) - 1:
        ax.axvline(current_x - 0.5, color='black', alpha=0.7, linestyle='--', linewidth=2)
ax.set_xlabel('Fit')
ax.set_ylabel('Uncertainty')
ax.set_title('Polynomial Coefficient Uncertainties')
plt.xticks(rotation=90)
plt.grid(".-")
plt.legend(loc='best', fontsize=f)
plt.savefig("Lab 2/plots/OH_Coef_Uncerctainty.pdf")
plt.show()

# R^2 Values
print("----------")
print("R^2 Values")
print("----------")
r_array = []
for val in range(0,len(values)):
    total_square_sum = np.sum((values[val] - np.mean(values[val]))**2)  # total sum of squares
    res_square_sum = np.sum(residuals[val]**2)  # residual sum of squares
    r_squared = 1 - (res_square_sum / total_square_sum)  # R^2 value
    print("R^2 = {r}".format(r=r_squared))
    r_array.append(r_squared)

plt.suptitle("Linear Wavelength Solution")
plt.title("$R^2 = ${r}".format(r=r_array[0]),fontsize=f+3)
plt.errorbar(centroids, known_lines, yerr=rmse[0], fmt='o', label="Centroid Wavelength Mapping \n RMS Error {r}".format(r=rmse[0]), color="red")
plt.plot(pixels,lin(pixels))
plt.xlabel("Pixel Position (Y Axis)")
plt.ylabel("True Wavelength (Angstroms)")
plt.grid(".-")
plt.legend(loc = 'best', fontsize = f)
plt.savefig("Lab 2/plots/OH_Linear.pdf")
plt.show()

plt.suptitle("Quadratic Wavelength Solution")
plt.title("$R^2 = ${r}".format(r=r_array[1]),fontsize=f+3)
plt.errorbar(centroids, known_lines, yerr=rmse[1], fmt='o', label="Centroid Wavelength Mapping \n RMS Error {r}".format(r=rmse[1]), color="red")
plt.plot(pixels,quad(pixels))
plt.xlabel("Pixel Position (Y Axis)")
plt.ylabel("True Wavelength (A)")
plt.grid(".-")
plt.legend(loc = 'best', fontsize = f)
plt.savefig("Lab 2/plots/OH_Quadratic.pdf")
plt.show()

plt.suptitle("Cubic Wavelength Solution")
plt.title("$R^2 = ${r}".format(r=r_array[2]),fontsize=f+3)
plt.errorbar(centroids, known_lines, yerr=rmse[2], fmt='o', label="Centroid Wavelength Mapping \n RMS Error {r}".format(r=rmse[2]), color="red")
plt.plot(pixels,cub(pixels))
plt.xlabel("Pixel Position (Y Axis)")
plt.ylabel("True Wavelength (A)")
plt.grid(".-")
plt.legend(loc = 'best', fontsize = f)
plt.savefig("Lab 2/plots/OH_Cubic.pdf")
plt.show()

# All fits plotted and extended to check for overfititng
pixels_extended = np.arange(ymin-len(pixels),ymax+len(pixels))
plt.title("Wavelength Solutions Compared")
plt.plot(pixels_extended,lin(pixels_extended), color="orange", label="Linear: RMS Error {r}".format(r=np.round(rmse[0],2)) ,alpha=0.5, ls="--")
plt.plot(pixels_extended,quad(pixels_extended), color="blue", label="Quadratic: RMS Error {r}".format(r=np.round(rmse[1],2)), alpha=0.5, ls="--")
plt.plot(pixels_extended,cub(pixels_extended), color="green", label="Cubic: RMS Error {r}".format(r=np.round(rmse[2],2)), alpha=0.5, ls="--")
plt.scatter(centroids,known_lines, color="red", label="Centroid Wavelength Mapping")
plt.xlabel("Pixel Position (Y Axis)")
plt.ylabel("True Wavelength")
plt.grid(".-")
plt.legend(loc="best",fontsize=f)
plt.savefig("Lab 2/plots/OH_Sol_Comparison.pdf")
plt.show()

# Plot Residuals
bound = np.max(linear_residuals) + 10
fig, ax = plt.subplots(3, 1, figsize=(6, 7), sharex=True)
ax[0].scatter(centroids, residuals[0], color="red",label="Linear")
ax[0].set_title("Residuals")
ax[0].set_ylim(-bound,bound)
ax[0].set_xlim(ymin,ymax)
ax[0].grid(True)
ax[0].legend(loc = 'best', fontsize = f)
ax[1].scatter(centroids, residuals[1], color="teal", label="Quadratic")
ax[1].set_ylim(-bound,bound)
ax[1].set_xlim(ymin,ymax)
ax[1].legend(loc = 'best', fontsize = f)
ax[1].grid(True)
ax[2].scatter(centroids, residuals[2], color="green", label="Cubic")
ax[2].set_xlabel("Pixel Number")
ax[2].set_xlim(ymin,ymax)
ax[2].set_ylim(-bound,bound)
ax[2].legend(loc = 'best', fontsize = f)
ax[2].grid(True)
plt.tight_layout()
plt.subplots_adjust(top=0.95)  
plt.savefig("Lab 2/plots/OH_Sol_Residuals.pdf")
plt.show()

plt.title("Quadratic Wavelength Solution")
plt.plot(quad(pixels),intensities,color="teal", label="Intensity")
plt.xlabel("Wavelength (A)")
plt.ylabel("Intensity")
plt.grid()
plt.legend(loc = 'best', fontsize = f)
plt.savefig("Lab 2/plots/OH_Quad_Spectrum.pdf")
plt.show()

# FE2 Emission Occurs at the third to last peak by visual inspection. I will create another
# DS9 projection which includes the FE2 emission for analysis, to rigorously find the peak using 
# the centroid 

plt.title("Spectrum Including Fe II Emission")
plt.plot(emission_pixels,emission_intensities,color="teal", label="Intensity")
plt.xlabel("Pixel Position (Y Axis)")
plt.ylabel("Intensity")
plt.grid()
plt.legend(loc = 'best', fontsize = f)
plt.savefig("Lab 2/plots/supernova_pixels.pdf")
plt.show()

plt.title("Fe II Emission Wavelength Solution")
plt.plot(quad(emission_pixels),emission_intensities,color="teal", label="Intensity")
plt.xlabel("Wavelength (A)")
plt.ylabel("Intensity")
plt.grid()
plt.legend(loc = 'best', fontsize = f)
plt.savefig("Lab 2/plots/supernova_wavelengths.pdf")
plt.show()

# Now applying wavelength solution to the new data, first I will find the centroid of the FE2 
# emission peak. 

emission_peak, properties = sp.signal.find_peaks(emission_intensities,height=0.97*np.max(emission_intensities))

# Calculate Centroid Boundaries
range_value=4
right_emission_boundary = emission_pixels[emission_peak[0]]+range_value 
left_emission_boundary = emission_pixels[emission_peak[0]]-range_value

# Plotting Centroid Bounds
plt.title("Fe II Emission Centroid Boundary")
plt.xlabel("Pixel Position")
plt.xlim(ymin,ymax)
plt.ylim(0,np.max(emission_intensities+10))
plt.ylabel("Intensity")
plt.plot(emission_pixels,emission_intensities,label="Intensity")
plt.scatter(emission_pixels[emission_peak[0]],emission_intensities[emission_peak[0]], ls='', marker="x", color="red", label="Peaks")
plt.vlines(right_emission_boundary,color="red",ymin=0,ymax=np.max(emission_intensities),alpha=0.5, label="Right Boundary: +{p} pixels".format(p=range_value))
plt.vlines(left_emission_boundary,color="green",ymin=0,ymax=np.max(emission_intensities),alpha=0.5, label="Left Boundary: -{p} pixels".format(p=range_value))
plt.legend(loc = 'best', fontsize = f)
plt.grid(".-")
plt.savefig("Lab 2/plots/supernova_boundary.pdf")
plt.show()

# Calculate Centroids
peak_centroid = [centroid(emission_pixels[emission_peak[i]-range_value:emission_peak[i]+range_value], emission_intensities[emission_peak[i]-range_value:emission_peak[i]+range_value]) for i in range(len(emission_peak))]


# Find wavelength of FE2 Emission
observed_wavelength = quad(peak_centroid[0])
print("---------")
print("Wavelength of FE2 Emission is: {w} Angstroms".format(w=observed_wavelength))

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(10,6), 
                               gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1})
ax0.plot(emission_pixels, emission_intensities, color = 'purple', label = 'Data')
ax0.axvline(peak_centroid[0], color = 'teal', label = 'Centroid Pixel Number: {p}'.format(p=np.round(peak_centroid[0],2)))
ax0.plot(emission_pixels[emission_peak], emission_intensities[emission_peak], color = 'coral', ls = '', marker = 'x', ms = 5, label = 'Peaks')
ax0.set_ylabel('Intensity')
ax0.legend(loc = 'best', fontsize = f)
ax0.grid(ls = '-.')
ax1.plot(emission_pixels[emission_peak], emission_pixels[emission_peak] - peak_centroid, ls = '', marker = '.', color = 'coral')
ax1.set_xlabel('Pixel Position (Y Axis)')
ax1.set_ylabel('Residuals')
ax1.grid(ls = '-.')
ax0.set_xlim(ymin, ymax)
ax0.set_ylim(0, np.max(emission_intensities))
ax1.set_ylim(-1, 1)
ax0.set_title("Fe II Emission Pixel Position")
plt.savefig("Lab 2/plots/supernova_centroid.pdf")
plt.show()

# Plotting The Peak Centroid w/ Wavelengths
fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(10,6), 
                               gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1})
ax0.plot(quad(emission_pixels), emission_intensities, color = 'purple', label = 'Data')
ax0.axvline(quad(peak_centroid[0]), color = 'teal', label = 'Wavelength: {w}'.format(w=np.round(quad(peak_centroid[0]),2)))
ax0.plot(quad(emission_pixels[emission_peak]), emission_intensities[emission_peak], color = 'coral', ls = '', marker = 'x',
         ms = 5, label = 'Peaks')
ax1.plot(quad(emission_pixels[emission_peak]), emission_pixels[emission_peak] - peak_centroid, ls = '', marker = '.', color = 'coral')
ax1.set_xlabel('Wavelength (A)')
ax0.set_ylabel('Intensity')
ax1.set_ylabel('Residuals')
ax0.legend(loc = 'best', fontsize = f)
ax0.grid(ls = '-.')
ax1.grid(ls = '-.')
ax0.set_xlim(quad(ymin), quad(ymax))
ax0.set_ylim(0,np.max(emission_intensities))
ax1.set_ylim(-1,1)
ax0.set_title("Fe II Centroid Wavelength")
plt.savefig("Lab 2/plots/supernova_cen_wavelength.pdf")
plt.show()

rmse_peak = np.sqrt(np.mean((emission_pixels[emission_peak] - peak_centroid)**2))

# Error Propogation 
# Sources of error are initial centroid placement, polynomial fitting, and peak wavelength centroid placement

# df/dx for the two centroid placements were found by running the program many times and finding the rate of change of the velocity per
# unit pixel. 
derivative = polyder(quad)
peak_derivative = derivative(peak_centroid[0])
propogated_error = np.sqrt((44.15)**2*rmse_centroid+(peak_derivative)**2*rmse[1]+(43.81)**2*rmse_peak)

print("Initial centroid rmse ", rmse_centroid)
print("df/dx", 44.15)
print("df/dy", peak_derivative)
print("rmse solution", rmse[1])
print("peak centroid rmse ", rmse_peak)
print("df/dz", 43.81)

# Calculating Velocity of Emission Using Doppler Shift Formula
intrensic_wavelength = 1.6439981 # microns
intrensic_wavelength_angstroms = intrensic_wavelength*1e4 # angstroms
velocity = c.c * (observed_wavelength-intrensic_wavelength_angstroms)/intrensic_wavelength_angstroms
print("Propogated Error is {p} km/s".format(p=propogated_error))

print("Velocity of FE2 Emission is {v} km/s".format(v=velocity*1e-3))