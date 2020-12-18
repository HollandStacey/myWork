import phot_pipe as pp
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from quick_image import *
from tqdm import tqdm,trange

from FITS_tools.hcongrid import hcongrid

prefix = 'PPP_M27'
targname = 'M27'
fwhm_max = 4.0
dpix_max = 50
dtheta_max = 5

paths = pp.get_paths(obstype='PPP',targname=targname)

Lfiles,Lct = pp.get_files(prefix=prefix,tag='Luminance')
info = pp.check_headers(Lfiles)
i1, = np.where((~np.isnan(info['fwhm']) & (~np.isnan(info['temp']))))
inds, = np.where((info['fwhm'][i1]*0.61 < fwhm_max)&((info['temp'][i1]<-28)&(info['temp'][i1]>-32)))
Lfiles = [Lfiles[i] for i in i1[inds]]

Lum,Lh = pp.all_stack(Lfiles,fwhm_max=fwhm_max,dpix_max=dpix_max,dtheta_max=dtheta_max,write=False)

Rfiles,Rct = pp.get_files(prefix=prefix,tag='Red')
info = pp.check_headers(Rfiles)
i1, = np.where((~np.isnan(info['fwhm']) & (~np.isnan(info['temp']))))
inds, = np.where((info['fwhm'][i1]*0.61 < fwhm_max)&((info['temp'][i1]<-28)&(info['temp'][i1]>-32)))
Rfiles = [Rfiles[i] for i in i1[inds]]

Red,Rh = pp.all_stack(Rfiles,fwhm_max=fwhm_max,dpix_max=dpix_max,dtheta_max=dtheta_max,write=False)

Gfiles,Gct = pp.get_files(prefix=prefix,tag='Green')
info = pp.check_headers(Gfiles)
i1, = np.where((~np.isnan(info['fwhm']) & (~np.isnan(info['temp']))))
inds, = np.where((info['fwhm'][i1]*0.61 < fwhm_max)&((info['temp'][i1]<-28)&(info['temp'][i1]>-32)))
Gfiles = [Gfiles[i] for i in i1[inds]]

Green,Gh = pp.all_stack(Gfiles,fwhm_max=fwhm_max,dpix_max=dpix_max,dtheta_max=dtheta_max,write=False)

Bfiles,Bct = pp.get_files(prefix=prefix,tag='Blue')
info = pp.check_headers(Bfiles)
i1, = np.where((~np.isnan(info['fwhm']) & (~np.isnan(info['temp']))))
inds, = np.where((info['fwhm'][i1]*0.61 < fwhm_max)&((info['temp'][i1]<-28)&(info['temp'][i1]>-32)))
Bfiles = [Bfiles[i] for i in i1[inds]]

Blue,Bh = pp.all_stack(Bfiles,fwhm_max=fwhm_max,dpix_max=dpix_max,dtheta_max=dtheta_max,write=False)

newL = hcongrid(Lum,Lh,Gh)
newR = hcongrid(Red,Rh,Gh)
newB = hcongrid(Blue,Bh,Gh)

fits.writeto(paths['output']+'Luminance.fits', np.float32(newL), Gh)
fits.writeto(paths['output']+'Red.fits', np.float32(newR), Gh)
fits.writeto(paths['output']+'Green.fits', np.float32(Green), Gh)
fits.writeto(paths['output']+'Blue.fits', np.float32(newB), Gh)

cmd = 'convert '+paths['output']+'Luminance.fits -format TIFF -depth 16 '+paths['output']+'Luminance.tif'
doit = os.system(cmd)

cmd = 'convert '+paths['output']+'Red.fits -format TIFF -depth 16 '+paths['output']+'Red.tif'
doit = os.system(cmd)

cmd = 'convert '+paths['output']+'Green.fits -format TIFF -depth 16 '+paths['output']+'Green.tif'
doit = os.system(cmd)

cmd = 'convert '+paths['output']+'Blue.fits -format TIFF -depth 16 '+paths['output']+'Blue.tif'
doit = os.system(cmd)

