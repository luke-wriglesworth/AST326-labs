from astropy.io import fits
import numpy as np

# Open Science file
science_file = fits.open("Part 3/GroupS/Moon_10S_20231026_21_50PM.fit")
science = np.array(science_file[0].data)

# Open Dark Files 
dark1 = fits.open("Part 3/GroupS/DARK_10S_20231026_21_58.fit")
dark2 = fits.open("Part 3/GroupS/DARK_10S_20231026_22_02.fit")
dark3 = fits.open("Part 3/GroupS/DARK_10S_20231026_22_03.fit")

# Open Flat Files 
flat1 = fits.open("Part 3/GroupS/Flat_5S_01.fit")
flat2 = fits.open("Part 3/GroupS/Flat_5S_02.fit")
flat3 = fits.open("Part 3/GroupS/Flat_5S_03.fit")

# Median Combine Flats
flat1_array = np.array(flat1[0].data)
flat2_array = np.array(flat2[0].data)
flat3_array = np.array(flat3[0].data)
flats_array = np.array([flat1_array,flat2_array,flat3_array])
master_flat = np.median(flats_array, axis=0)

# Median Combine Darks
dark1_array = np.array(dark1[0].data)
dark2_array = np.array(dark2[0].data)
dark3_array = np.array(dark3[0].data)
darks_array = np.array([dark1_array,dark2_array,dark3_array])
master_dark = np.median(darks_array, axis=0)

# My darks are all 10 second exposures, so assuming the noise scales linearly I 
# can just divide them by 2 for 5 seconds worth of noise 

master_dark_5S = master_dark / 2

# Steps
# 1. Subtract Master Dark From Science File which also includes Bias Noise 
science_sub_dark = science - master_dark_5S

# 2. Subtract dark from flat 
flat_sub_dark = master_flat - master_dark_5S

# 3. Divide 1 by 2
final_array  = science_sub_dark / flat_sub_dark

# Export processed FIT file 
final_fit = fits.PrimaryHDU(final_array)
final_fit.writeto('Part 3/GroupS/processed_fits/Moon_10S_Processed.fit')
