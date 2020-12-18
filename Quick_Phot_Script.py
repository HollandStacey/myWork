import phot_pipe as pp
from quick_image import *
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy import units as u

# Important information
obstype = 'Cal'
targname = 'SA38326'

# Get all the dates that the above target was observed
dates = pp.get_dates(targname=targname)
# Pick one date to play around with
date = dates[0]
# RA and Dec of the source
ra = '18:47:40.516'
dec = '45:24:39.37'

# If you run this on your local computer, make sure you have your paths
# set up in phot_pipe.py.
# If you are running this as student on bellerophon, it should work fine.
paths = pp.get_paths(targname=targname, obstype=obstype)

# Get calibration files
cals = pp.get_cal_frames(date=date, targname=targname, obstype=obstype, write=True, archive=False)

# Find all the files on a given date
sfiles,sct = pp.get_files(date=date, prefix=targname,tag='-V')
# Pick one file to play with
sfile = sfiles[0]

# Read image and header from file
image,header = read_image(sfile,plot=False)

# Calibrate image
calim = pp.calibrate_image(image,header, cals)
# Look at calibrated image
display_image(calim)

# Turn the coordinates into an x and y pixel
w = wcs.WCS(header)
c = SkyCoord(ra,dec, unit=(u.hour, u.deg))
xpos, ypos = wcs.utils.skycoord_to_pixel(c, w)

# Specify sky annulus
skyrad = [40,50]

# This will determine the optimal aperture, but also find the total counts and total flux
op_dict = pp.optimal_aperture(calim,header,x=xpos,y=ypos,skyrad=skyrad,plot=True,recenter=True)

print(op_dict['totflux'])

# Once you get this working, simply loop through all the dates and all the files on each date
# This is what we call a nested loop.

# Have fun!!!

