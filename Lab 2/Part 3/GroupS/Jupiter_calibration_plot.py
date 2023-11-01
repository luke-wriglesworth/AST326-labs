# Imports 
import numpy as np
from numpy import polyfit
from numpy import poly1d
from numpy import polyder
import scipy as sp
import scipy.constants as c
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Loading Neon Sample
projection = np.loadtxt("Lab 2/Part 3/GroupS/processed_fits/neon_spectrum.dat")

known_lines = [420.067,585.249,692.947, 
               703.241, 724.517,743.89,
               750.387,763.511]

# DS9 Y Pixel Ranges
xmin = 0
xmax = 1392

f=11 # font size

pixels = np.arange(xmin,xmax)
jupiter_pixels = np.arange(xmin,xmax)
intensities = projection[:,1]

# Jupiter Spectrum
jupiter = np.loadtxt("Lab 2/Part 3/GroupS/processed_fits/Jupiter_spectrum.dat")
jupiter_intensities = jupiter[:,1] + abs(np.min(jupiter[:,1]))

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
plt.savefig("Lab 2/plots/obs_neon_spectrum.pdf")
plt.show()

# Peak Finding 
min_height = 0.07 * np.max(intensities)
initial_peaks, properties = sp.signal.find_peaks(intensities, height=min_height)
peaks = np.delete(initial_peaks,[2,3,4,11]) # Removing Peaks

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
for i, peak in enumerate(peaks):
    if i % 1 == 0:
        plt.annotate(str(known_lines[i])+" nm", (pixels[peak], intensities[peak]), 
                     xytext=(-120, 5), textcoords='offset points', arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle="angle,angleA=0,angleB=90,rad=10"), 
                     label="Wavelengths (nm)", fontsize = f-3)
plt.grid(ls = '-')
plt.legend(loc = 'best', fontsize = f)
plt.savefig("Lab 2/plots/obs_neon_peaks.pdf")
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
plt.savefig("Lab 2/plots/obs_boundaries.pdf")
plt.show()

# Calculate Centroids
centroids = [centroid(pixels[peaks[i]-range_value:peaks[i]+range_value], intensities[peaks[i]-range_value:peaks[i]+range_value]) for i in range(len(peaks))]
centroid_residuals = np.array(centroids) - np.array(peaks)
rmse_centroid = np.sqrt(np.mean(centroid_residuals**2))

# Plotting The Centroids and Residuals
fig = plt.figure(figsize=(10,6))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace = 0.1)
ax0 = plt.subplot(gs[0])
ax0.set_xlim(1100,np.max(pixels))
ax0.set_title("Peak Centroids")
ax1 = plt.subplot(gs[1], sharex = ax0)
ax1.set_ylim(-1,1)
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
plt.savefig("Lab 2/plots/obs_cen_residuals.pdf")
plt.show()

#fitting



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

# Calculate Uncertainty For Coefficients
linear_uncertainty = np.sqrt(np.diag(linear_cov))
quad_uncertainty = np.sqrt(np.diag(quad_cov))
cub_uncertainty = np.sqrt(np.diag(cub_cov))
uncertainty_coefficients = [linear_uncertainty,quad_uncertainty,cub_uncertainty]

print("Uncertainty of Coefficients")
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
    ax.bar(x_values, uncertainty, width=0.3, label="{f}".format(f=fit_names[i]))
    current_x += len(uncertainty)
    if i < len(uncertainty_coefficients) - 1:
        ax.axvline(current_x - 0.5, color='r', linestyle='-')  # subtract 0.5 to position the line between fits
ax.set_xlabel('Fit')
ax.set_ylabel('Uncertainty')
ax.set_title('Uncertainties of Polynomial Coefficients for Each Fit')
plt.xticks(rotation=90)
plt.grid(".-")
plt.legend(loc='best', fontsize=f)
plt.savefig("Lab 2/plots/obs_coef_uncertainty.pdf")
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

plt.suptitle("Linear Fit")
plt.title("$R^2 = ${r}".format(r=r_array[0]),fontsize=f+3)
plt.errorbar(centroids, known_lines, yerr=rmse[0], fmt='o', label="Centroid Wavelength Mapping \n RMS Error {r}".format(r=rmse[0]), color="red")
plt.plot(pixels,lin(pixels))
plt.xlabel("Pixel Position (X Axis)")
plt.ylabel("True Wavelength (nm)")
plt.grid(".-")
plt.legend(loc = 'best', fontsize = f)
plt.savefig("Lab 2/plots/obs_linear.pdf")
plt.show()

plt.suptitle("Quadratic Fit")
plt.title("$R^2 = ${r}".format(r=r_array[1]),fontsize=f+3)
plt.errorbar(centroids, known_lines, yerr=rmse[1], fmt='o', label="Centroid Wavelength Mapping \n RMS Error {r}".format(r=rmse[1]), color="red")
plt.plot(pixels,quad(pixels))
plt.xlabel("Pixel Position (X Axis)")
plt.ylabel("True Wavelength (nm)")
plt.grid(".-")
plt.legend(loc = 'best', fontsize = f)
plt.savefig("Lab 2/plots/obs_quadratic.pdf")
plt.show()

plt.suptitle("Cubic Fit")
plt.title("$R^2 = ${r}".format(r=r_array[2]),fontsize=f+3)
plt.errorbar(centroids, known_lines, yerr=rmse[2], fmt='o', label="Centroid Wavelength Mapping \n RMS Error {r}".format(r=rmse[2]), color="red")
plt.plot(pixels,cub(pixels))
plt.xlabel("Pixel Position (X Axis)")
plt.ylabel("True Wavelength (nm)")
plt.grid(".-")
plt.legend(loc = 'best', fontsize = f)
plt.savefig("Lab 2/plots/obs_cubic.pdf")
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
plt.savefig("Lab 2/plots/obs_neon_residuals.pdf")
plt.show()

# All fits plotted and extended to check for overfititng
pixels_extended = np.arange(xmin-len(pixels),xmax+len(pixels))
plt.title("All Fits Compared")
plt.plot(pixels_extended,lin(pixels_extended), color="orange", label="Linear: RMS Error {r}".format(r=np.round(rmse[0],2)) ,alpha=0.5, ls="--")
plt.plot(pixels_extended,quad(pixels_extended), color="blue", label="Quadratic: RMS Error {r}".format(r=np.round(rmse[1],2)), alpha=0.5, ls="--")
plt.plot(pixels_extended,cub(pixels_extended), color="green", label="Cubic: RMS Error {r}".format(r=np.round(rmse[2],2)), alpha=0.5, ls="--")
plt.scatter(centroids,known_lines, color="red", label="Centroid Wavelength Mapping")
plt.xlabel("Pixel Position (Y Axis)")
plt.ylabel("True Wavelength")
plt.grid(".-")
plt.legend(loc="best",fontsize=f)
plt.savefig("Lab 2/plots/obs_sol_comparison.pdf")
plt.show()

plt.title("Neon Lamp Wavelength Solution")
plt.plot(lin(pixels),intensities,color="teal", label="Intensity")
plt.xlim(lin(1),lin(xmax+10))
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity")
plt.grid()
plt.legend(loc = 'best', fontsize = f)
plt.savefig("Lab 2/plots/obs_neon_solution.pdf")
plt.show()

wavelength_1 = 470.50436018390144
wavelength_2 = 470.1599428930929
dw = wavelength_1 - wavelength_2

# Plotting Jupiter pixels
plt.figure(figsize=(12,6))
plt.title("Jupiter Spectrum")
plt.plot(pixels,jupiter_intensities,color="teal", label="Intensity")
plt.xlim(1,xmax)
plt.xlabel("Pixels")
plt.ylabel("Intensity")
plt.grid()
plt.legend(loc = 'best', fontsize = f)
plt.savefig("Lab 2/plots/obs_jupiter_pixels.pdf")
plt.show()

# Plotting Jupiter wavelength
plt.figure(figsize=(12,6))
plt.title("Jupiter Wavelength Solution")
plt.plot(lin(pixels),jupiter_intensities,color="teal", label="Intensity")
plt.xlim(300,lin(xmax))
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity")
plt.grid()
plt.legend(loc = 'best', fontsize = f)
plt.savefig("Lab 2/plots/obs_jupiter_spectrum.pdf")
plt.show()

# Error Propogation due to centroid placements and Fitting
derivative = polyder(lin)

# Obtained by running the program multiple times and finding the rate of change of wavelength / pixel at 500 pixels 
# by shifting the centroid placements.
centroid_df_dx = 0.34441729080856476
fit_df_dx = derivative(500)
propogated_error = np.sqrt(centroid_df_dx**2*rmse_centroid*fit_df_dx**2*rmse[0])
print("Propogated Error is {e}".format(e=propogated_error))

