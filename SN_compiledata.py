'''
Purpose: To compile all supernova and non-supernova images, filter out
the supernova images, inject fake supernova into some of the non-supernova
images then cut them into 10x10 images and finally, condolidate all supernova,
fake-supernova, and non-supernova images into one file/folder/directory.

Author: Cyrus Leung & Teddy Rosenbaum (and Dr. Swift :)

Created: 10/19/2020

Last Updated: 10/19/2020
'''

# Import useful libraries
#import os
from tqdm import tqdm
import phot_pipe as pp
# import astropy as ap
import pandas as pd
import numpy as np
from quick_image import read_image

files, fct = pp.get_files(prefix='SNS_')
files = np.array(files)

# Loop through each file
fwhm_vec = [] ; bg_vec = [] ; obj = [] ; date_obs = []
pbar = tqdm(desc = 'Analyzing SNS files', total = fct, unit = 'files')
for fname in files:
    image, header = read_image(fname)
    info = pp.do_sextractor(image, header)
    fwhm_vetted = pp.sigmaRejection(info['FWHM_WORLD']*3600.0)
    fwhm_vec = np.append(fwhm_vec,np.median(fwhm_vetted))
    bg_vec = np.append(bg_vec,np.median(info['BACKGROUND'])/header['EXPTIME'])
    obj = np.append(obj, header['OBJECT'].split('_')[-1])
    date_obs = np.append(date_obs, fname.split('/')[-2])
    pbar.update(1)
pbar.close()

# Compile dataframe
columns = ['FileName','Target','FWHM','Background','Obs. Date']
df = pd.DataFrame(columns = columns)
df['FileName'] = files
df['Target'] = obj
df['FWHM'] = fwhm_vec
df['Background'] = bg_vec
df['Obs. Date'] = date_obs

# Export csv
path = pp.get_paths(obstype="SNS")['output']+'SNS_imagesinfo.csv'
df.to_csv(r'/home/student/SN/SN_imageinfo.csv', index=False)
# Count all the SNS files that exist in the Archive


# Compile a list of which targets have been observed and how many
# times they have been observed



# Where are the supernova images?
# Where are the rest of the images?

# Add fake supernova to non-supernova images

# Add images to dataframe

# Export dataframe as a csv file
''' Code for just adding date
import pandas as pd
from quick_image import read_image
import numpy as np
import phot_pipe as pp

files, fct = pp.get_files(prefix='SNS_')
files = np.array(files)
df = pd.read_csv('/home/student/SN/SN_imageinfo.csv')
date_obs = []
for fname in files:
    image, header = read_image(fname)
    date_obs = np.append(date_obs, header['DATE-OBS'])
df['Obs. Date'] = date_obs
df.to_csv(r'/home/student/SN/SN_imageinfo_wdates.csv', index=False)
'''


    
