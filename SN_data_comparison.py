'''
Author: Holland Stacey
    
23 October 2020
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from astropy.time import Time
import pdb, pickle
import phot_pipe as pp
from astropy.coordinates import SkyCoord
from astropy import units as u
from statsmodels.robust import mad

def SN_data_comparison(plotband='i',offset = 4,target='2020hvf',target_ra = '11:21:26.44',target_dec = '03:00:52.92'):
    '''
    Graph Thacher reduction of Thacher data vs. UCSC reduction of Thacher data of a given target in one band
        
    Keywords:
    ---------
    offset (float): global shift in magnitude
        
        
        '''
    
    #Get and read thacher data
    fname = '/home/student/SNe/SN'+target+'/SN'+target+'_photometry.pck'
    thacher_df = pickle.load( open( fname, "rb" ) )
    thacher_data = thacher_df[plotband]
    thacher_mag = thacher_data['mag'] + offset
    thacher_magerr = thacher_data['mag_err']
    thacher_bjd = thacher_data['bjd']
    
    
    #Read UCSC json and print out a dataframe with mag, magerr, and date
    coords = SkyCoord(target_ra,target_dec,unit=(u.hour,u.degree))
    UCSC_dir = '/home/student/SNe/SN'+target+'/'
    sc_data = pd.read_json(UCSC_dir+target+'_data.json')
    
    # Unpack UCSC data
    for d in sc_data[target]['photometry']:
        instrument = d['fields']['instrument']
        corrected_instument = instrument.split(' ')[0]
        if corrected_instument == u'Thacher':
            sc_mag = []
            sc_magerr = []
            sc_bjd = []
            for sd in d['data']:
                if sd['fields']['mag_err'] < 0.5:
                    b = sd['fields']['band']
                    band = sd['fields']['band'].split('-')[-1].replace(' ','')
                    if band == plotband:
                        t = Time(sd['fields']['obs_date'],format='isot',scale='utc')
                        sc_t_bjd = pp.jd_to_bjd(t.jd, coords)
                        sc_bjd = np.append(sc_bjd,sc_t_bjd)
                        stuff = sd['fields']['mag'] + offset
                        sc_mag = np.append(sc_mag,stuff)
                        sc_magerr = np.append(sc_magerr,sd['fields']['mag_err'])



    sc_inds = np.argsort(sc_bjd)
    sc_bjd = sc_bjd[sc_inds]
    sc_mag = sc_mag[sc_inds]
    sc_magerr = sc_magerr[sc_inds]

    thacher_inds = np.argsort(thacher_bjd)
    thacher_bjd = thacher_bjd[thacher_inds]
    thacher_mag = thacher_mag[thacher_inds]
    thacher_magerr = thacher_magerr[thacher_inds]

    bjd_final = []
    thacher_mag_final = []
    thacher_magerr_final = []
    sc_mag_final = []
    sc_magerr_final = []

    # Build new arrays for data that agrees in time within 10 min threshold
    for i in range(len(sc_bjd)):
        bjd = sc_bjd[i]
        diff = np.min(np.abs(bjd-thacher_bjd))*1440
        diff_ind = np.argmin(np.abs(bjd-thacher_bjd))
        if diff < 10:
            bjd_final = np.append(bjd_final,thacher_bjd[diff_ind])
            sc_mag_final = np.append(sc_mag_final, sc_mag[i])
            sc_magerr_final = np.append(sc_magerr_final, sc_magerr[i])
            thacher_mag_final = np.append(thacher_mag_final, thacher_mag[diff_ind])
            thacher_magerr_final = np.append(thacher_magerr_final, thacher_magerr[diff_ind])


    
    '''
    print(bjd_final)
        
    print(sc_mag_final)
    print(sc_magerr_final)
        
    print(thacher_mag_final)
    print(thacher_magerr_final)
    '''
    
    mag =  thacher_mag_final - sc_mag_final
    mag_err = np.sqrt(thacher_magerr_final**2 + sc_magerr_final**2)
    
    
    return bjd_final-np.min(bjd_final),mag,mag_err


data = {}
offsets = [0,2,3,4]
bands = ['g','r','i','z']
dev = []
meds = []
for i in range(len(bands)):
    band = bands[i]
    offset = offsets[i]2
    day,mag,mag_err = SN_data_comparison(plotband=band,offset=offset)
    data[band+' mag'] = mag
    data[band+' magerr'] = mag_err
    data[band+' time'] = day
    dev = np.append(dev,mad(mag))
    meds = np.append(meds,np.median(mag))

yrange = np.max(dev)*10


colors = ['green','red','brown','black']

plt.ion()
plt.figure(1)
plt.clf()

dy = 0.15
for i in range(len(bands)):
    color=colors[i]
    band = bands[i]
    plt.errorbar(data[band+' time'],data[band+' mag']+i*dy,yerr=data[band+' magerr'],fmt='o',color=color,label=band)
    plt.axhline(y=i*dy,linestyle='--',color='r')

plt.ylim(-yrange,yrange+len(bands)*dy)
plt.legend(loc='best')
plt.xlabel('Time - $t_0$ (days)')
plt.ylabel(r'$\Delta m$ + offset')
plt.title("Thacher - UCSC for SN"+"2020hvf",fontsize=16)


# sort in terms of bjd, use intersect
#final_err = np.sqrt(sc_magerr**2+thacher_magerr**2)

#np.intersect1d(sc_bjd, thacher_bjd, assume_unique=False, return_indices=True)
#np.sort(np.array(thacher_bjd))

#maybe write a for loop to get rid of non-matching data points
#for i in thacher_bjd:
#if i
#np.subtract(thacher_mag, sc_mag)

plt.annotate('Median g ='+np.array2string(meds[0]),xy=[0.40,0.85],
             xycoords='figure fraction',horizontalalignment='right')
plt.annotate('Median r ='+np.array2string(meds[1]),xy=[0.40,0.81],
             xycoords='figure fraction',horizontalalignment='right')
plt.annotate('Median i ='+np.array2string(meds[2]),xy=[0.40,0.77],
             xycoords='figure fraction',horizontalalignment='right')
plt.annotate('Median z ='+np.array2string(meds[3]),xy=[0.40,0.73],
             xycoords='figure fraction',horizontalalignment='right')
