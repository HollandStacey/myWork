##########################################
from tqdm import tqdm, trange
import sys
sys.path.append('/home/student/python/photometry')
sys.path.append('/home/student/python/utils')
from quick_image import *
import pickle
import pandas as pd
import numpy as np
import os
from astropy.io import fits
import time
from datetime import datetime
import matplotlib.pyplot as plt
from ast import literal_eval
import phot_pipe as pp
from panstarrs_query import *
import FITS_tools.hcongrid
import shutil
from astropy.wcs import WCS

def run_diff_dophot(galaxy, thefile,tag='_diff',redo = False):
    #Make tag include time stamp

    print '=========================='
    thefile = glob.glob(thefile)
    if len(thefile) == 0:
        print "Input file: ",thefile," does not exist! Exiting..."
        return
    else:
        ##################################
        #Read image files
        #template  = '/home/student/SNS/templates/stacked/'+galaxy+'_stacked.fit'
        print 'Running Dophot...'
        print 'Galaxy:  ', galaxy
        thefile = thefile[0]

        template_name = thefile.split('/')[-1]

        timestamp = '_'+template_name.split('_')[1].split('-')[1]
        #images = glob.glob('/home/student/SNS/'+galaxy+'/*.fit')
        #diffim, tim, th, sim, sh = diff(template,images[0])
        print 'Input file: ', thefile
        done_files = [os.path.basename(x) for x in glob.glob('/home/student/DoPHOT_C/*')]
        if 'psf_out_'+galaxy+timestamp+tag+'.fits' in done_files and template_name in done_files and redo==False:
            print 'DoPHOT results Already exist! Exiting...'
            return

        ###################################
        print 'Modifying DoPhot parameters file...'
        #Move stuff to the dophot directory and modify parameters file (pm)
        dophot_dir = '/home/student/diff_dophot/'
        #if thefile not in glob.glob(dophot_dir+'*'):
        #    shutil.copy(thefile,dophot_dir)
        with open(dophot_dir+'pm') as f:
            pm = f.readlines()

        pm[12] = "IMAGE_IN = '"+template_name+"'               Input image name.\n"
        #pm[12] = "IMAGE_IN = '"+galaxy+"_stacked.fit'               Input image name.\n"
        pm[13] =  "IMAGE_OUT = 'image_out_"+galaxy+timestamp+tag+".fits'    Output image name.\n"
        pm[15] =  "OBJECTS_OUT = 'obj_out_"+galaxy+timestamp+tag+"'         Output object list file name.\n"
        pm[17] = "=ERRORS_OUT = 'errors_out_"+galaxy+timestamp+tag+"       Errors on fit to be output if output type is internal\n"
        pm[18] =  "SHADOWFILE_OUT = 'shad_out_"+galaxy+timestamp+tag+"'           Output shadow file name.\n"
        pm[19] = "EMP_SUBRAS_OUT = 'psf_out_"+galaxy+timestamp+tag+".fits'       Empirical PSF subraster.\n"
        pm[22] =  "=COVARS_OUT = 'covar_out_"+galaxy+timestamp+tag+"'    File to contain covariance matrices\n"

        textfile = open(dophot_dir+'pm', 'w')
        for i in range(len(pm)):
            textfile.write(pm[i])
        textfile.close()
        ################################
        #Run DoPHOT
        print "Running DoPHOT..."
        os.chdir(dophot_dir)
        os.system('cdophot pm')
        #if show_results:
    return


"""
dophot_dir = '/home/student/diff_dophot/'
files = glob.glob(dophot_dir+'diff*')
for f in files:
	run_diff_dophot('NGC0865',f)
"""











