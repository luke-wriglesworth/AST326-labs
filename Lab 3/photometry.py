import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

import os
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
import matplotlib.patches as patches
import photutils as ph
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import numpy as np 
import scipy as sp
from scipy.optimize import curve_fit
from datetime import datetime
from scipy.stats import powerlaw

# Default plot parameters
plt.rcParams["figure.figsize"] = (10,6)
plt.rcParams['font.size'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Returns the intensity array from a .fits file and corrects overflowing pixels
def read_file(filename):
    # Need to fix bright pixels wrapping around into negative values.
    image = fits.open("AST325-326-SN-short/{f}".format(f=filename))[0].data.astype(int)
    # -1e3 will keep the sub-zero noise but will still remove the very negative values that 
    # are obviously due to overflow. 
    overflow_magnitude = np.abs(image[image < -500])
    max_value = 2**15-1
    # unwrap negative pixels to positive
    image[image < -500] = (max_value-overflow_magnitude) + max_value
    return image

def get_header(filename):
    header = fits.open("AST325-326-SN-short/{f}".format(f=filename))[0].header
    return header

def get_headers(namelist):
    headers = []
    for i, name in enumerate(namelist):
        current_header = get_header(name)
        headers.append(current_header)
    return headers

# Puts the image names in order and returns a new ordered namelist
def sort_names(names):
    name_list = np.loadtxt((names), dtype=str)
    return np.sort(name_list)

# Create a numpy array of images, based on a name list
def image_array(namelist):
    images = []
    for i, name in enumerate(namelist):
        images.append(read_file(name))
    return np.array(images)
    
# View all images in array, also can plot histograms to check for overflow
def view_all(images, names,xcen,ycen,box,zoom=False):
    headers = get_headers(names)
    for i, image in enumerate(images):
        w = WCS(headers[i])
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection=w)
        print(names[i])
        print(headers[i]['DATE-OBS'])
        #print("Max Value: {max}, Min Value: {min}".format(max=np.max(image),min=np.min(image)))
        if zoom:
            ax.set_xlim(xcen-box,xcen+box)
            ax.set_ylim(ycen-box,ycen+box)
        ax.imshow(image, vmin=-100, vmax = 1e2, origin = 'lower', cmap = 'inferno')
        plt.show()
        
        # # Plot histogram of entire image as one row to check for overflowing pixels
        # plt.figure(figsize=(12,5))
        # plt.hist(image.flatten(), bins = 'sturges', color = 'k', histtype='step')
        # plt.grid(alpha=0.5)
        # plt.yscale('log')
        # plt.xlabel("Pixel Value [ADU]", size = 16)
        # plt.ylabel("Number of Pixels", size = 16)
        # plt.show()
        
# View a single image
def view(image):
    plt.figure(figsize=(10,10))
    plt.imshow(image, vmin=-100, vmax = 1e2, origin = 'lower', cmap = 'inferno')
    plt.show()
        
# Calculates residuals of the images with the median image
def image_residuals(images,median):
    residuals = []
    for i, image in enumerate(images):
        residuals.append(image - median)
    return np.array(residuals)
    
def sum_residuals(residuals):
    errors = []
    for i, residual in enumerate(residuals):
        sum_square = np.sum(np.square(residual))
        errors.append(sum_square)
    return np.array(errors)
        
def median_filter_images(images, names, distances, upper, lower):
    
# def magnitude_filter_images(images, names,mag):
#     continue
    good = []
    good_i = []
    bad = []
    bad_i = []
    for i, image in enumerate(images):
        if not ((distances[i] >= upper) or (distances[i] <= lower)):
            good.append(image)
            good_i.append(i)
        else:
            bad.append(image)
            bad_i.append(i)
    print("{g} good images out of {t}".format(g=len(good),t=len(images)))
    return np.array(good), np.array(good_i), np.array(bad), np.array(bad_i)
        
def centroid(x, y, image, box):
    x = int(x)
    y = int(y)
    box = int(box)
    x_num = 0
    x_den = 0
    y_num = 0
    y_den = 0
    sub_image = image[y-box:y+box, x-box:x+box]
    for i, row in enumerate(sub_image):
        for j, intensity in enumerate(row):
            x_num += j * intensity
            x_den += intensity
    for i, col in enumerate(sub_image.T):
        for j, intensity in enumerate(col):
            y_num += j * intensity
            y_den += intensity
    x_centroid = x_num / x_den + x - box
    y_centroid = y_num / y_den + y - box
    return x_centroid, y_centroid
     
def aperture(image, xcenter, ycenter, radius, plot=False):
    ysize, xsize = image.shape
    y, x = np.mgrid[:ysize, :xsize]
    mask = ((x - xcenter)**2 + (y - ycenter)**2) <= radius**2
    aperture_sum = np.sum(image * mask)
    if plot:
        plt.xlim(xcenter-20,xcenter+20)
        plt.ylim(ycenter-20,ycenter+20)
        plt.imshow(image, vmin=-100, vmax = 1e2, origin='lower', cmap='inferno')
        plt.imshow(mask, origin='lower', cmap='inferno', alpha=0.5)
        plt.show()
    return aperture_sum

def annulus(image, xcenter, ycenter, inner_radius, outer_radius, plot=False):
    ysize, xsize = image.shape
    y, x = np.mgrid[:ysize, :xsize]
    r_squared = (x - xcenter)**2 + (y - ycenter)**2
    mask = (r_squared <= outer_radius**2) & (r_squared >= inner_radius**2)
    annulus_sum = np.sum(image * mask)
    num_pixels = np.sum(mask)
    level_per_pixel = annulus_sum / num_pixels if num_pixels else 0
    result = level_per_pixel * num_pixels
    
    # Uncertainty
    sigma_bg = np.std(image*mask)
    annulus_uncertainty = np.sqrt(num_pixels * sigma_bg)
    
    if plot:
        plt.xlim(xcenter-20,xcenter+20)
        plt.ylim(ycenter-20,ycenter+20)
        plt.imshow(image, vmin=-100, vmax = 1e2, origin='lower', cmap='inferno')
        plt.imshow(mask, origin='lower', cmap='gray', alpha=0.3)
        plt.show()
    return result, annulus_uncertainty

def gaus(x,a,x0,sigma, c):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + c

# Given a range of sample images, returns the radius to use for the aperture & annulus, based on 3
# times the value of the std of the fitted gausian to each image. 
def sn_radius(sample_range1,sample_range2,x,y,box_sn):
    gauss_std1 = []
    gauss_std2 = []
    for i,image in enumerate(sample_range1):
        sn_x_cen,sn_y_cen = centroid(x,y,image,box_sn)
        cutout_sn = image[int(sn_y_cen-box):int(sn_y_cen+box), int(sn_x_cen-box):int(sn_x_cen+box)]
        x_sn_gauss, y_sn_gauss = np.arange(cutout_sn.shape[0]), cutout_sn[cutout_sn.shape[0]//2, :]
        popt_sn, pcov_sn = curve_fit(gaus, x_sn_gauss, y_sn_gauss, p0=[np.max(cutout_sn), cutout_sn.shape[0]//2, 10, 0])
        gauss_std1.append(abs(popt_sn[2]))
    for i,image in enumerate(sample_range2):
        sn_x_cen,sn_y_cen = centroid(x,y,image,box_sn)
        cutout_sn = image[int(sn_y_cen-box):int(sn_y_cen+box), int(sn_x_cen-box):int(sn_x_cen+box)]
        x_sn_gauss, y_sn_gauss = np.arange(cutout_sn.shape[0]), cutout_sn[cutout_sn.shape[0]//2, :]
        popt_sn, pcov_sn = curve_fit(gaus, x_sn_gauss, y_sn_gauss, p0=[np.max(cutout_sn), cutout_sn.shape[0]//2, 10, 0])
        gauss_std2.append(abs(popt_sn[2]))
    avg_std1 = np.mean(gauss_std1)
    avg_std2 = np.mean(gauss_std2)
    radius_sn = 3*np.mean((avg_std1+avg_std2)/2)
    sn_r1 = radius_sn + radius_sn / 3
    sn_r2 = sn_r1 + sn_r1 / 2
    print(radius_sn)
    return radius_sn, sn_r1, sn_r2
    
# Sort image names in order
sorted_names = sort_names("AST325-326-SN-filelist.txt")

# Get numpy array of sorted images
images = image_array(sorted_names)

# View images in order 
#view_all(images, sorted_names)

# Need to remove bad images...
median = np.median(images)
image_residuals = image_residuals(images,median)

# Finding the sum-squared distance from the median of the images
distances = sum_residuals(image_residuals)
median_error = np.median(distances)
mad = sp.stats.median_abs_deviation(distances)

upper_bound = 1.3e11
lower_bound = 5.2e9

# # # Plot the distances to find outliers and set a threshold for images to keep. 
# image_numbers = np.arange(0,len(sorted_names))
# plt.title("Distances From Mean of Median")
# plt.xlabel("Image Number")
# plt.ylabel("Distance (Relative to median of images)")
# plt.yscale("log")
# plt.axhline(median_error, color = "Red", ls="--", label="Median Error")
# plt.axhline(upper_bound, color="green", label="Upper Error Threshold: {e}".format(e=upper_bound))
# plt.axhline(lower_bound, color="brown", label="Lower Error Threshold: {e}".format(e=lower_bound))
# plt.scatter(image_numbers,distances,label="Distance", s=0.5)
# plt.grid(".-")
# plt.legend(loc="best")
# plt.show()

# Find bad/good images 
good_imgs, good_indices, bad_imgs, bad_indices = median_filter_images(images, sorted_names, distances, upper_bound, lower_bound)
good_names = sorted_names[good_indices]

#view_all(good_imgs,good_names,0,0,0)
#view_all(bad_imgs,sorted_names[bad_indices])
#aperture(good_imgs[0],250,250,20,plot=True)

# Reference Star 1: (00:56:49.70, -37:01:38.31)
# Reference Star 2: (00:56:46.43, -37:02:29.50)
# Reference Star 3: (00:56:58.27,-36:58:16.60)

ref_1 = SkyCoord('00:56:49.70 -37:01:38.31', unit=('hourangle', 'deg'))
ref_2 = SkyCoord('00:56:46.43 -37:02:29.50', unit=('hourangle', 'deg'))
ref_3 = SkyCoord('00:56:58.27 -36:58:16.60', unit=('hourangle', 'deg'))
sn = SkyCoord('00:57:03.19 -37:02:23.6', unit=('hourangle', 'deg'))

box=15
box_sn=10
measurement_headers = []
aperture_measurement_ref1 = []
aperture_measurement_ref2 = []
aperture_measurement_ref3 = []
aperture_measurement_sn = []
annulus_measurement_ref1 = []
annulus_measurement_ref2 = []
annulus_measurement_ref3 = []
annulus_measurement_sn = []

# Annulus uncertainties
annulus_uncertainty_ref1 = []
annulus_uncertainty_ref2 = []  
annulus_uncertainty_ref3 = []
annulus_uncertainty_sn = []


measured_indices_in_good_imgs = []

sample1 = good_imgs[150:160]
sample2 = good_imgs[250:260]

#view_all(sample1,good_names[150:160],100,125,box_sn,zoom=True)
#print("Second sample...")
#view_all(sample2,good_names[250:260],100,125,box_sn,zoom=True)

radius_sn, sn_r1, sn_r2 = sn_radius(sample1,sample2,100,125,box_sn)

for i,image in enumerate(good_imgs):
    header = get_header(good_names[i])
    w = WCS(header)
    # Pixel positions of objects
    x1, y1 = w.all_world2pix(ref_1.ra.deg, ref_1.dec.deg, 1)
    x2, y2 = w.all_world2pix(ref_2.ra.deg, ref_2.dec.deg, 1)
    x3, y3 = w.all_world2pix(ref_3.ra.deg, ref_3.dec.deg, 1)
    #x_sn,y_sn = w.all_world2pix(sn.ra.deg, sn.dec.deg, 1)
    x_sn,y_sn = 100,125

    # Centroids of objects
    x1_cen,y1_cen = centroid(x1,y1,image,box)
    x2_cen,y2_cen = centroid(x2,y2,image,box)
    x3_cen,y3_cen = centroid(x3,y3,image,box)
    sn_x_cen,sn_y_cen = centroid(x_sn,y_sn,image,box_sn)
    
    # I'm checking the intensity of reference star 2 and 1 to skip any images with unreasonably low values, since
    # some images with missing stars made it past the initial median filter.     
    initial_measurement_ref2 = aperture(image,x2_cen,y2_cen,5)
    initial_measurement_ref1 = aperture(image,x1_cen,y1_cen,5)
    initial_measurement_ref3 = aperture(image,x3_cen,y3_cen,5)
    initial_measurement_sn = aperture(image,sn_x_cen,sn_y_cen,5)
    threshold = 2000
    if initial_measurement_ref2 > threshold and initial_measurement_ref1 > threshold and initial_measurement_ref3 > threshold:        
        # Create a cutout around the centroid positions
        cutout_ref1 = image[int(y1_cen-box):int(y1_cen+box), int(x1_cen-box):int(x1_cen+box)]
        cutout_ref2 = image[int(y2_cen-box):int(y2_cen+box), int(x2_cen-box):int(x2_cen+box)]
        cutout_ref3 = image[int(y3_cen-box):int(y3_cen+box), int(x3_cen-box):int(x3_cen+box)]

        # Fit a Gaussian to the central row of the cutout to find aperture radius
        x_ref1, y_ref1 = np.arange(cutout_ref1.shape[0]), cutout_ref1[cutout_ref1.shape[0]//2, :]
        x_ref2, y_ref2 = np.arange(cutout_ref2.shape[0]), cutout_ref1[cutout_ref2.shape[0]//2, :]
        x_ref3, y_ref3 = np.arange(cutout_ref3.shape[0]), cutout_ref1[cutout_ref3.shape[0]//2, :]

        popt_ref1, pcov_ref1 = curve_fit(gaus, x_ref1, y_ref1, p0=[np.max(cutout_ref1), cutout_ref1.shape[0]//2, 10, 0])
        popt_ref2, pcov_ref2 = curve_fit(gaus, x_ref2, y_ref2, p0=[np.max(cutout_ref2), cutout_ref2.shape[0]//2, 10, 0])
        popt_ref3, pcov_ref3 = curve_fit(gaus, x_ref3, y_ref3, p0=[np.max(cutout_ref3), cutout_ref3.shape[0]//2, 10, 0])

        # # Plot the central row and the fitted Gaussian
        # plt.figure(figsize=(10, 5))
        # plt.plot(x, y, 'x', color='k', label='Central Row')
        # plt.plot(x, gaus(x, *popt), color='r', alpha=0.3, label=f'Best-fit Gaussian (STD {round(popt[2], 1)})')
        # plt.legend()
        # plt.show()
        
        
        # lets use 3 times the STD of the gaussian for the aperture radius. 
        radius_ref1 = abs(3*popt_ref1[2])
        radius_ref2 = abs(3*popt_ref2[2])
        radius_ref3 = abs(3*popt_ref3[2])

        # annulus radii
        ref1_r1 = radius_ref1 + radius_ref1 / 3
        ref1_r2 = ref1_r1 + ref1_r1 / 2
        ref2_r1 = radius_ref2 + radius_ref2 / 3
        ref2_r2 = ref2_r1 + ref2_r1 / 2
        ref3_r1 = radius_ref3 + radius_ref3 / 3
        ref3_r2 = ref3_r1 + ref3_r1 / 2

        aperture1 = aperture(image, x1_cen, y1_cen, radius_ref1)
        aperture2 = aperture(image, x2_cen, y2_cen, radius_ref2)
        aperture3 = aperture(image, x3_cen, y3_cen, radius_ref3)
        aperture_sn = aperture(image, sn_x_cen, sn_y_cen, radius_sn)
        
        # Check again that we are measuring valid values, including supernova data
        if not np.isnan(aperture1) and not np.isnan(aperture2) \
        and not np.isnan(aperture3) and not np.isnan(aperture_sn):

            if aperture1 > threshold and aperture2 > threshold \
            and aperture3 > threshold and aperture_sn > 0:
                        
                # Record index
                measured_indices_in_good_imgs.append(i)
                
                # Record measurements
                aperture_measurement_ref1.append(aperture(image, x1_cen,y1_cen,radius_ref1))
                annulus_measurement_ref1.append(annulus(image,x1_cen,y1_cen,ref1_r1,ref1_r2))
                aperture_measurement_ref2.append(aperture(image, x2_cen,y2_cen,radius_ref2))
                annulus_measurement_ref2.append(annulus(image,x2_cen,y2_cen,ref2_r1,ref2_r2))
                aperture_measurement_ref3.append(aperture(image, x3_cen,y3_cen,radius_ref3))
                annulus_measurement_ref3.append(annulus(image,x3_cen,y3_cen,ref3_r1,ref3_r2))
                aperture_measurement_sn.append(aperture(image,sn_x_cen,sn_y_cen,radius_sn,plot=False))
                annulus_measurement_sn.append(annulus(image,sn_x_cen,sn_y_cen,sn_r1,sn_r2,plot=False))
                
                # Get annulus uncertainties
                annulus_uncertainty_ref1.append(annulus(image,x1_cen,y1_cen,ref1_r1,ref1_r2,plot=False)[1])
                annulus_uncertainty_ref2.append(annulus(image,x2_cen,y2_cen,ref2_r1,ref2_r2,plot=False)[1])
                annulus_uncertainty_ref3.append(annulus(image,x3_cen,y3_cen,ref3_r1,ref3_r2,plot=False)[1])
                annulus_uncertainty_sn.append(annulus(image,sn_x_cen,sn_y_cen,sn_r1,sn_r2,plot=False)[1])
                
                # # Draw centroid target lines
                # fig = plt.figure(figsize=(10,10))
                # ax = fig.add_subplot(111, projection=w)
                # # Zoom on ref 1
                # ax.set_xlim(x1_cen-100,x1_cen+100)
                # ax.set_ylim(y1_cen-100,y1_cen+100)
                # ax.axvline(x1_cen,color="green")
                # ax.axhline(y1_cen,color="green")
                # ax.imshow(image, vmin=-100, vmax = 1e2, origin = 'lower', cmap = 'inferno')
                # plt.show()

measured_indices = good_indices[measured_indices_in_good_imgs]

# Indices of measured images that correspond to good_imgs array. 
print("{f} images out of {i} remaining after final filter".format(f=len(measured_indices),i=len(images)))
ref1_intensities = np.abs(np.array(aperture_measurement_ref1) - np.array(annulus_measurement_ref1)[:, 0])
ref2_intensities = np.abs(np.array(aperture_measurement_ref2) - np.array(annulus_measurement_ref2)[:, 0])
ref3_intensities = np.abs(np.array(aperture_measurement_ref3) - np.array(annulus_measurement_ref3)[:, 0])
sn_signal = np.array(aperture_measurement_sn)
sn_noise = np.array(annulus_measurement_sn)[:, 0]
sn_intensities = np.abs(sn_signal - sn_noise)
sn_signal_noise = sn_signal / sn_noise

# Source Uncertainties
gain = 4
sn_source_uncertainty = np.sqrt(sn_intensities/gain)
ref1_src_uncertainty = np.sqrt(ref1_intensities/gain)
ref2_src_uncertainty = np.sqrt(ref2_intensities/gain)
ref3_src_uncertainty = np.sqrt(ref3_intensities/gain)

# Calculate total intensity uncertainties
sn_intensity_uncertainty = np.square(np.sqrt(np.square(sn_source_uncertainty) + np.square(annulus_uncertainty_sn)))
ref1_intensity_uncertainty = np.square(np.sqrt(np.square(ref1_src_uncertainty) + np.square(annulus_uncertainty_ref1)))
ref2_intensity_uncertainty = np.square(np.sqrt(np.square(ref2_src_uncertainty) + np.square(annulus_uncertainty_ref2)))
ref3_intensity_uncertainty = np.square(np.sqrt(np.square(ref3_src_uncertainty) + np.square(annulus_uncertainty_ref3)))

# Find where s/n ratio goes above 3 for first detection threshold
detected_sn_3 = np.where(sn_signal_noise > 3)[0][0]
first_detection_fn = sorted_names[measured_indices[detected_sn_3]]
date_detected_header = get_header(first_detection_fn)['DATE-OBS']
date_detected = datetime.fromisoformat(date_detected_header)
sn_month_day = "{m}-{d}".format(m=date_detected.month,d=date_detected.day)

# Calculate magnitudes
ref3_mag = 15.61 # Uncertainty 0.02
ref2_mag = -2.5*np.log10(ref2_intensities/ref3_intensities) + ref3_mag
ref1_mag = -2.5*np.log10(ref1_intensities/ref3_intensities) + ref3_mag
sn_mag = 2.5*np.log10(ref3_intensities/sn_intensities) + ref3_mag

# Propogate reference star uncertainties to get SN magnitude uncertainties
ref3_mag_uncertainty = 0.02
sigma_ref1 = (2.5/np.log(10)) * (sn_intensity_uncertainty/sn_intensities)
sigma_ref2 = (2.5/np.log(10)) * (ref2_intensity_uncertainty/ref2_intensities)
sn_magnitude_uncertainty = np.sqrt(sigma_ref1**2 + sigma_ref2**2 + ref3_mag_uncertainty**2)

plt.title("Supernova Magnitude")
plt.xlabel("Image Number")
plt.ylabel("Magnitude")
plt.ylim(17.5, 24)
plt.scatter(measured_indices, sn_mag, color="blue", label="Supernova Magnitude", s=2)
plt.errorbar(measured_indices, sn_mag, yerr=sn_magnitude_uncertainty, fmt='none', color='red', label='Uncertainty', lw=0.5, alpha=0.3, ls='--')
plt.axvline(measured_indices[detected_sn_3], color="red", ls="--", label="SN > 3: {i}".format(i=date_detected))
plt.gca().invert_yaxis()
plt.grid(".-")
plt.legend()
plt.show()

# Get the dates of all measured images
month_days = []
full_dates = []
for index in measured_indices:
    # Get date
    current_name = sorted_names[index]
    current_header = get_header(current_name)
    current_date = datetime.fromisoformat(current_header['DATE-OBS'])
    month_days.append("{month}-{date}".format(month=current_date.month, date=current_date.day))
    full_dates.append(current_date)

# Bin the images by day
days = []
measurements = []
uncertainties = []
full_names = []
intensities = []
for i in range(len(month_days)):
    day = month_days[i]
    full_name = full_dates[i]
    measurement = sn_mag[i]
    intensity_current = sn_intensities[i]
    uncertainty = sn_magnitude_uncertainty[i]
    if day not in days:
        days.append(day)
        measurements.append([measurement])
        uncertainties.append([uncertainty])
        full_names.append([full_name])
        intensities.append([intensity_current])
    else:
        index = days.index(day)
        measurements[index].append(measurement)
        uncertainties[index].append(uncertainty)
        full_names[index].append(full_name)
        intensities[index].append(intensity_current)

# Plot the binned magnitudes with uncertainties
plt.title("Supernova Magnitudes")
plt.xlabel("Days")
plt.ylabel("Magnitude")
plt.ylim(17.5, 24.5)
plt.xlim(-20,20)
plt.xticks(rotation=45)
plt.gca().invert_yaxis()
for i in range(len(days)):
    day = days[i]
    measurement_list = measurements[i]
    uncertainty_list = uncertainties[i]
    measurements_in_day = len(measurement_list) * [day]
    plt.errorbar(measurements_in_day, measurement_list, yerr=uncertainty_list, fmt='o', markersize=3, capsize=3, alpha=0.3, lw=0.5, color='blue', ls='--', ecolor='darkblue')
    plt.scatter(measurements_in_day, measurement_list, color='blue', s=5)
detection = plt.axvline(sn_month_day, color="red", ls="--", label="First Detection: {i}".format(i=date_detected))
mag_label = lines.Line2D([], [], color='blue', marker='o', markersize=3, ls='None', label="Magnitude Measurement")
uncertainty_label = lines.Line2D([], [], color='darkblue', marker='o', markersize=3, ls='None', label="Uncertainty")
plt.legend(loc="lower right", fontsize=14, handles=[mag_label, uncertainty_label, detection])
plt.grid(".-")
plt.show()

# Calculate weighted mean of each day based on uncertainties, also calculate mean uncertainty
weighted_means = []
mean_uncertainties = []
for i in range(len(days)):
    measurement_list = measurements[i]
    uncertainty_list = uncertainties[i]
    weights = 1 / np.array(uncertainty_list)
    weighted_mean = np.sum(measurement_list * weights) / np.sum(weights)
    mean_uncertainty = np.mean(np.array(uncertainty_list))
    weighted_means.append(weighted_mean)
    mean_uncertainties.append(mean_uncertainty)

plt.title("Weighted Mean Magnitudes")
plt.xlabel("Days")
plt.ylabel("Magnitude")
plt.ylim(17.5, 24.5)
plt.xticks(rotation=45)
plt.gca().invert_yaxis()
plt.scatter(days, weighted_means, color='blue', label='Weighted Mean', s=7)
plt.errorbar(days, weighted_means, yerr=mean_uncertainties, fmt='none', color='green', label='Mean Uncertainty', lw=0.5, alpha=0.3, ls='--')
plt.axvline(sn_month_day, color="red", ls="--", label="First Detection: {i}".format(i=date_detected))
mag_label = lines.Line2D([], [], color='blue', marker='o', markersize=3, ls='None', label="Weighted Mean")
uncertainty_label = lines.Line2D([], [], color='green', marker='o', markersize=3, ls='None', label="Mean Uncertainty")
detection = plt.axvline(sn_month_day, color="red", ls="--", label="First Detection: {i}".format(i=date_detected))
plt.legend(loc="lower right", fontsize=14, handles=[mag_label, uncertainty_label, detection])
plt.grid(".-")
plt.show()

# Fit a polynomial to the weighted mean
poly2 = np.polyfit(np.arange(len(days)), weighted_means, 2)
poly3 = np.polyfit(np.arange(len(days)), weighted_means, 3)
poly4 = np.polyfit(np.arange(len(days)), weighted_means, 4)

# Evaluate magnitudes from polynomial
magnitudes4 = np.polyval(poly4, np.arange(len(days)))
magnitudes3 = np.polyval(poly3, np.arange(len(days)))
peak_day_index3 = np.argmin(magnitudes3)
peak_time_measurement = np.argmax(np.array(measurements[peak_day_index3]))
peak_time_name = full_names[peak_day_index3][peak_time_measurement]
peak_magnitude = magnitudes3[peak_day_index3]

date_detected_index = days.index(sn_month_day)
days_since_detection = [i - date_detected_index - 1 for i in range(len(days))]
days_since_detection_np = np.array(days_since_detection)

print("Peak Magnitude: {m}".format(m=magnitudes3[peak_day_index3]))

# Plot weighted mean along with binned data, and fit polynomials to the weighted mean.
fig, ax1 = plt.subplots()
fig.figsize=(12,9)
ax1.set_title("Supernova Magnitude", pad=5)
ax1.set_xlabel("Date")
ax1.set_ylabel("B-Band Magnitude")
ax1.set_ylim(17.5, 24.5)
ax1.set_xticks(np.arange(len(days)))
ax1.set_xticklabels(days, rotation=45)
ax1.invert_yaxis()
for i in range(len(days)):
    day = days[i]
    measurement_list = measurements[i]
    uncertainty_list = uncertainties[i]
    measurements_in_day = len(measurement_list) * [day]
    ax1.errorbar(measurements_in_day, measurement_list, yerr=uncertainty_list, fmt='o', markersize=3, capsize=3, alpha=0.3, lw=0.5, color='blue', ls='--', ecolor='darkblue')
    ax1.scatter(measurements_in_day, measurement_list, color='blue', s=5)
ax1.scatter(days, weighted_means, color='red', marker='x', label='Weighted Mean', s=18)
ax2 = ax1.twiny()
ax2.set_xlabel("Days Since Epoch of First Detection", fontsize=12)
ax2.set_xticks(days_since_detection_np)
ax2.set_xticklabels(days_since_detection, rotation=45)
ax2.xaxis.set_ticks_position('top')
ax1.plot(days, magnitudes3, color='orange', label='3rd Order Polynomial Fit', ls='--')
detection = ax1.axvline(sn_month_day, color="red", ls="--", label="Epoch of First Detection: {i}".format(i=date_detected))
peak_brightness_cubic = ax1.axvline(days[peak_day_index3], color="brown", ls="--")
mag_label = lines.Line2D([], [], color='blue', marker='o', markersize=3, ls='None', label="Magnitude Measurement")
uncertainty_label = lines.Line2D([], [], color='darkblue', marker='o', markersize=2, ls='None', label="Uncertainty")
weighted_mean_label = lines.Line2D([], [], color='red', marker='x', markersize=9, ls='None', label="Weighted Mean")
polynomial_label = lines.Line2D([], [], color='purple', ls='--', label="3rd Order Polynomial Fit")
plt.legend(loc="lower right", fontsize=14, handles=[mag_label, uncertainty_label, weighted_mean_label, polynomial_label, detection, peak_brightness_cubic])
plt.show()

# Power law chi squared fitting

# Plot normalized intensities
plt.title("Supernova Normalized Intensity")
plt.xlabel("Days")
plt.ylabel("Normalized Intensity")
plt.xlim(0,len(days)) 
plt.set_xticks(days_since_detection_np) 
plt.xticks(rotation=45)
#plt.gca().invert_yaxis()
for i in range(len(days)):
    day = days[i]
    intensity_list = intensities[i]
    measurements_in_day = len(intensity_list) * [day]
    plt.scatter(measurements_in_day, intensity_list, color='blue', s=5)
detection = plt.axvline(sn_month_day, color="red", ls="--", label="First Detection: {i}".format(i=date_detected))
intensity_label = lines.Line2D([], [], color='blue', marker='o', markersize=3, ls='None', label="Normalized Intesnity")
plt.legend(loc="lower right", fontsize=14, handles=[intensity_label])
plt.grid(".-")
plt.show()