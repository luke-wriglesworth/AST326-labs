import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
import matplotlib.patches as patches
import photutils as ph
import matplotlib.pyplot as plt
import numpy as np 
import scipy as sp

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
    overflow_magnitude = np.abs(image[image < -1e3])
    max_value = 2**15-1
    # unwrap negative pixels to positive
    image[image < -1e3] = (max_value-overflow_magnitude) + max_value
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
def view_all(images, names):
    headers = get_headers(names)    
    for i, image in enumerate(images):
        w = WCS(headers[i])
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection=w)
        print(names[i])
        print(headers[i]['DATE-OBS'])
        #print("Max Value: {max}, Min Value: {min}".format(max=np.max(image),min=np.min(image)))
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
    return np.array(good), good_i, np.array(bad), bad_i
        
        
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

upper_bound = median_error + 1.5*mad
lower_bound = median_error - 1.5*mad

# # Plot the distances to find outliers and set a threshold for images to keep. 
# image_numbers = np.arange(0,len(sorted_names))
# plt.title("Distances From Mean of Median")
# plt.xlabel("Image Number")
# plt.ylabel("Distance (Relative to median of images)")
# plt.yscale("log")
# plt.axhline(median_error, color = "Red", ls="--", label="Median Error")
# plt.axhline(upper_bound, color="green", label="Upper Error Threshold")
# plt.axhline(lower_bound, color="brown", label="Lower Error Threshold")
# plt.scatter(image_numbers,distances,label="Distance", s=0.5)
# plt.grid(".-")
# plt.legend(loc="best")
# plt.show()

# Find bad/good images 
good_imgs, good_indices, bad_imgs, bad_indices = median_filter_images(images, sorted_names, distances, upper_bound, lower_bound)

#view_all(good_imgs,sorted_names[good_indices])
#view_all(bad_imgs,sorted_names[bad_indices])


