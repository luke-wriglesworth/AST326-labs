# Imports 
import numpy as np
from numpy import polyfit
from numpy import poly1d
from numpy import polyder
import scipy as sp
import scipy.constants as c
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

neon_data = np.loadtxt("Lab 2/Part 1/Ne_calib.dat")

true_wavelengths = [540.056,576.441,582.015,585.249,588.189,594.483,597.553,602.00,607.433,
                         614.306,616.359,621.728,626.649,630.479,633.442,638.299,640.225,650.653,
                         653.288,655.895,667.828,671.204,692.947, 703.241,717.394,724.512,743.890,
                         747.244,748.887,753.577,754.404]
xmin = 0
xmax = 1024

# font size
f = 11
pixels = np.arange(xmin,xmax)

# Peak Finding For 1D Neon Lamp Source
min_height = 0.02 * np.max(neon_data)
initial_peaks, properties = sp.signal.find_peaks(neon_data, height=min_height)
print("Number of found peaks in neon lamp data {p}".format(p=len(initial_peaks)))

# Removing false peaks
peaks = np.delete(initial_peaks,[22])

plt.rcParams["figure.figsize"] = (10,6)
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['font.size'] = 16

plt.title("Neon Lamp Calibration Spectrum")
plt.xlabel("Pixel Number")
plt.ylabel("Relative Intensity")
plt.plot(pixels,neon_data,color="teal",linestyle="-", label="Relative Intensity")
plt.grid(".-")
plt.legend(loc = 'best', fontsize=f)
plt.savefig("Lab 2/plots/neon_calibration_pt1.pdf")
plt.show()


plt.title("Neon Lamp Spectrum With {n} Identified Peaks".format(n=len(peaks)))
plt.xlabel("Pixel Position")
plt.ylabel("Relative Intensity")
plt.plot(pixels,neon_data,label="Relative Intensity")
plt.plot(pixels[peaks],neon_data[peaks], ls='', marker="x", color="red", label="Identified Peak")
for i, peak in enumerate(peaks):
    if i % 6 == 0:
        plt.annotate(str(true_wavelengths[i])+" nm", (pixels[peak], neon_data[peak]), 
                     xytext=(-20, 60), textcoords='offset points', arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle="angle,angleA=0,angleB=90,rad=10"), 
                     label="Wavelengths (nm)", fontsize = f-3)
plt.grid(ls = '-')
plt.legend(loc='upper left', fontsize=f)
plt.savefig("Lab 2/plots/neon_peaks.pdf")
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
plt.title("Centroid Boundary Ranges")
plt.xlabel("Pixel Position")
plt.xlim(500,900)
plt.ylim(0,0.7*np.max(neon_data))
plt.ylabel("Relative Intensity")
plt.plot(pixels,neon_data,label="Relative Intensity")
plt.plot(pixels[peaks],neon_data[peaks], ls='', marker="x", color="red", label="Identified Peak")
plt.vlines(right_line_points,color="red",ymin=0,ymax=0.3*np.max(neon_data),alpha=0.5, label="Right Boundary: +{p} pixels".format(p=range_value))
plt.vlines(left_line_points,color="green",ymin=0,ymax=0.3*np.max(neon_data),alpha=0.5, label="Left Boundary: -{p} pixels".format(p=range_value))
plt.legend(loc='best',fontsize=f)
plt.grid(".-")
plt.savefig("Lab 2/plots/neon_boundaries.pdf")
plt.show()

# Calculate Centroids and residuals
centroids = [centroid(pixels[peaks[i]-range_value:peaks[i]+range_value], neon_data[peaks[i]-range_value:peaks[i]+range_value]) for i in range(len(peaks))]
centroid_residuals = np.array(centroids) - np.array(peaks)

# Plotting The Centroids and Residuals
fig = plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace = 0.0)
ax0 = plt.subplot(gs[0])
ax0.set_title("Centroids and Residuals")
ax0.set_xlim(500,900)
ax1 = plt.subplot(gs[1], sharex = ax0)
ax0.plot(pixels, neon_data, color = 'purple', label = 'Data')
ax0.plot(pixels[peaks], neon_data[peaks], color = 'coral', ls = '', marker = 'x',
         ms = 5, label = 'Peaks')
[ax0.axvline(c, color = 'teal', alpha = 0.5, label = 'Centroids') if c == centroids[0]
 else ax0.axvline(c, color = 'teal', alpha = 0.5) for c in centroids]
ax1.plot(pixels[peaks], pixels[peaks] - centroids, ls = '', marker = '.', color = 'coral')
ax1.set_xlabel('Pixels')
ax1.set_ylabel('Residual')
ax0.set_ylabel('Relative Intensity')
plt.setp(ax0.get_xticklabels(), visible = False)
ax0.legend(loc = 'best', fontsize = 10)
ax1.set_ylim(-2,2)
ax0.grid(ls = '-.')
ax1.grid(ls = '-.')
plt.savefig("Lab 2/plots/neon_centroids.pdf")
plt.show()

# root mean square error for centroids
rmse_centroid = np.sqrt(np.mean(centroid_residuals**2))
print("Centroid RMS Error: {r}".format(r=rmse_centroid))

# Finding the fits 
linear_fit, linear_cov = polyfit(centroids,true_wavelengths,1,full=False, cov=True)
lin = poly1d(linear_fit)
quadratic_fit, quad_cov = polyfit(centroids,true_wavelengths,2,full=False, cov=True)
quad = poly1d(quadratic_fit)
cubic_fit, cub_cov = polyfit(centroids,true_wavelengths,3,full=False, cov=True)
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
linear_residuals = np.array(true_wavelengths) - np.array(linear_values_cen)
quad_residuals = np.array(true_wavelengths) - np.array(quad_values_cen)
cub_residuals = np.array(true_wavelengths) - np.array(cub_values_cen)
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
ax.set_title("Polynomial Coefficient Uncertainties")
plt.xticks(rotation=90)
plt.grid(".-")
plt.legend(loc='best', fontsize=f)
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
plt.errorbar(centroids, true_wavelengths, yerr=rmse[0], fmt='o', label="Centroid Wavelength Mapping \n RMS Error {r}".format(r=rmse[0]), color="red")
plt.plot(pixels,lin(pixels))
plt.xlabel("Pixel Position (X Axis)")
plt.ylabel("Wavelength (nm)")
plt.grid(".-")
plt.legend(loc = 'best', fontsize = f)
plt.savefig("Lab 2/plots/neon_linear.pdf")
plt.show()

plt.suptitle("Quadratic Wavelength Solution")
plt.title("$R^2 = ${r}".format(r=r_array[1]),fontsize=f+3)
plt.errorbar(centroids, true_wavelengths, yerr=rmse[1], fmt='o', label="Centroid Wavelength Mapping \n RMS Error {r}".format(r=rmse[1]), color="red")
plt.plot(pixels,quad(pixels))
plt.xlabel("Pixel Position (X Axis)")
plt.ylabel("Wavelength (nm)")
plt.grid(".-")
plt.legend(loc = 'best', fontsize = f)
plt.savefig("Lab 2/plots/neon_quadratic.pdf")
plt.show()

plt.suptitle("Cubic Wavelength Solution")
plt.title("$R^2 = ${r}".format(r=r_array[2]),fontsize=f+3)
plt.errorbar(centroids, true_wavelengths, yerr=rmse[2], fmt='o', label="Centroid Wavelength Mapping \n RMS Error {r}".format(r=rmse[2]), color="red")
plt.plot(pixels,cub(pixels))
plt.xlabel("Pixel Position (X Axis)")
plt.ylabel("Wavelength (nm)")
plt.grid(".-")
plt.legend(loc = 'best', fontsize = f)
plt.savefig("Lab 2/plots/neon_cubic.pdf")
plt.show()

# All fits plotted and extended to check for overfititng
pixels_extended = np.arange(xmin-len(pixels),xmax+len(pixels))
plt.title("Wavelength Solutions Compared")
plt.plot(pixels_extended,lin(pixels_extended), color="orange", label="Linear: RMS Error {r}".format(r=np.round(rmse[0],2)) ,alpha=0.5, ls="--")
plt.plot(pixels_extended,quad(pixels_extended), color="blue", label="Quadratic: RMS Error {r}".format(r=np.round(rmse[1],2)), alpha=0.5, ls="--")
plt.plot(pixels_extended,cub(pixels_extended), color="green", label="Cubic: RMS Error {r}".format(r=np.round(rmse[2],2)), alpha=0.5, ls="--")
plt.scatter(centroids,true_wavelengths, color="red", label="Centroid Wavelength Mapping")
plt.xlabel("Pixel Position (Y Axis)")
plt.ylabel("Wavelength")
plt.grid(".-")
plt.legend(loc="best",fontsize=f)
plt.savefig("Lab 2/plots/neon_sol_comparison.pdf")
plt.show()

# Plot Residuals
bounds = 5
fig, ax = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
ax[0].scatter(centroids, linear_residuals, color="red")
ax[0].set_title("Linear Fit Residuals")
ax[0].set_ylabel("Residuals")
ax[0].set_ylim(-bounds,bounds)
ax[0].grid(True)
ax[1].scatter(centroids, quad_residuals, color="teal")
ax[1].set_title("Quadratic Fit Residuals")
ax[1].set_ylabel("Residuals")
ax[1].set_ylim(-bounds,bounds)
ax[1].grid(True)
ax[2].scatter(centroids, cub_residuals, color="green")
ax[2].set_title("Cubic Fit Residuals")
ax[2].set_xlabel("Pixel Number")
ax[2].set_ylabel("Residuals")
ax[2].set_ylim(-bounds,bounds)
ax[2].grid(True)
plt.tight_layout()
plt.subplots_adjust(top=0.95)  
plt.savefig("Lab 2/plots/neon_sol_residuals.pdf")
plt.show()

# Blackbody Calculations
bb_data = np.loadtxt("Lab 2/Part 1/Group_S_BB.dat")
pixels = np.arange(len(bb_data))

plt.title("Raw Data of Unkown Blackbody")
plt.xlabel("Pixel Number")
plt.ylabel("Relative Intensity")
plt.plot(pixels,bb_data,color="teal",linestyle="--", label="Relative Intensity")
plt.grid(".-")
plt.legend(loc = 'best', fontsize=f)
plt.savefig("Lab 2/plots/raw_blackbody.pdf")
plt.show()

# Using linear fit to calculate wavelengths
plt.title("Wavelength Spectrum of Blackbody")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Relative Intensity")
plt.plot(linear_values_all, bb_data, color="teal",linestyle="-")
plt.grid(".-")
plt.savefig("Lab 2/plots/blackbody_solution.pdf")
plt.show()

# Finding Peak Intensity Wavelength -> 628.4285223741275nm
max_wavelength = linear_values_all[np.argmax(bb_data)]
print("Wavelength of Highest Intensity is : {w}nm".format(w=max_wavelength))

# Derivative of fitting for error propogation
derivative = polyder(lin)
peak_derivative = derivative(np.argmax(bb_data))

# Errors to propogate
# 1. Centroid placement
# 2. Polynomial Fitting
# 3. Wien's law has no uncertainty since the peak wavelength is from an ideal blackbody which we know
# exactly. 

# df/dx for centroid placement is 1.73 Kelvin / 1 pixel, based on running the program many times
# uncertainty for centroid placement is RMS error

# df/dx for polynomial is the derivative of the polynomial 
# uncertainty is RMS error of residuals 

propogated_error = np.sqrt((1.73)**2*rmse_centroid + (peak_derivative)**2*rmse[0])
print("Propogated Error is {e} Kelvin".format(e=propogated_error))

# Now using Wien's displacement law to find temperature
wavelength_bb_meters = max_wavelength * 10**(-9)
print("Temperature of blackbody is: {t} Kelvin".format(t = c.Wien / wavelength_bb_meters))