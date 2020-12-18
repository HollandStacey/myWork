#System tools
import os,pickle,io,requests,string,urlparse,pdb,inspect,socket,glob,sys
#Utilities
import numpy as np
import pandas as pd
from scipy.odr import ODR, Model, Data, RealData
from length import length
from FITS_tools.hcongrid import hcongrid
from recursive_glob import recursive_glob
from tqdm import tqdm,trange
from statsmodels import robust
import scipy.optimize as opt
from scipy.special import erfinv
from collections import OrderedDict
from statsmodels.stats.diagnostic import normal_ad
import scipy

# Astropy stuff
from astropy.time import Time
from astropy import units as u
#Plotting stuff
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#Import from Thacher's photometry pipeline
import phot_pipe as pp
from Monitoring_utils import *
#--------Some Targets-------#
# Mrk501:      '16 53 52.13', '+39 45 36.2'
# PKS1510-089: '15 12 50.533', '-09 05 58.99'
#
#
#

"""
Created by Alejandro Wilcox in 2019 for simple Monitoring of any target
TODO: Make automatic creation of target_info for any source
os intersecting
global testing
Descriptions and commenting
More useful functions and plot_night
"""
def new_monitor(targname = None, ra = None, dec = None, Mag = None):
    if targname == None or ra == None or dec == None:
        print("All fields must be filled!!!")
    #Mag should eb dealt with, now manual, but should not be.
    paths = pp.get_paths(obstype = "Monitoring", targname = targname)
    try:
        os.mkdir(paths['output']+"plots/")
        os.mkdir(paths['output']+"plots/updates/")
        os.mkdir(paths['output']+"plots/lightcurves/")
    except:
        print "Error in making one of the directories! One of them probably already exists!"

    #Set up summary files and target_info pck
    pp.make_summary_file(obstype='Monitoring',targname=targname)
    make_target_info_pck(targname,ra,dec,Mag = Mag)
    paths,names,refs,targ_info,obs,obs_dates = setup(targname)
    make_parsed_photometry_pck(targname)

def do_phot(targname = None):
    paths,names,refs,targ_info,obs,obs_dates = setup(targname)

    dirfiles = os.listdir(paths['output'])
    phot_dates = [int(dirfiles[i]) for i in range(len(dirfiles)) if dirfiles[i][0:1] == '2']
    phot_dates = [phot_dates[i] for i in range(len(phot_dates)) if paths['output']+str(phot_dates[i])+'/'+"photometry_"+str(phot_dates[i])+'.pck' is not None]
    phot_dates.sort()

    new_dates = [str(obs_dates[i]) for i in range(len(obs_dates)) if obs_dates[i] not in phot_dates]
    if len(new_dates) == 0:
        print "No new photometry to be done since ", phot_dates[-1], "for " + targname
    else:
        print "Performing photometry on the following days for " + targname
        print new_dates

    for date in new_dates:
        pp.night_phot(date,obstype='Monitoring',targname=targname,rotated_flat=True)
    print "Finished photometry for all new dates"
    return


def old_parse_photometry(targname=None, flags = ['saturation','snr','airmass','fwhm','temperature'], output = False):
    paths,names,refs,targ_info,obs,obs_dates = setup(targname)
    #Load pickle and find last processed date
    all_bands = pickle.load(open(paths['output']+targname+'_parsed_photometry.pck','rb'))
    if len(all_bands[all_bands.keys()[0]]['bjd']) == 0:
        parsed_end_date = 0
    else:
        parsed_end_date = Time(int(all_bands[all_bands.keys()[0]]['bjd'][-1][0]+.5),format='jd',scale='utc').utc
        parsed_end_date.format = 'iso'
        list = parsed_end_date.value.split(' ')[0].split('-')
        parsed_end_date = list[0]+list[1]+list[2]
    print "Latest parsed date for " +targname+ ": ", parsed_end_date

    dirfiles = os.listdir(paths['output'])
    phot_dates = [int(dirfiles[i]) for i in range(len(dirfiles)) if dirfiles[i][0:1] == '2']
    if int(parsed_end_date) == max(phot_dates):
            print targname+"_parsed_photometry is already up-to-date with completed photometry!"
            return all_bands
    #New dates to parse
    new_dates = [str(phot_dates[i]) for i in range(len(phot_dates)) if phot_dates[i] > int(parsed_end_date)]
    new_dates.sort()
    print "Parsing photometry from the following days for "+targname, new_dates

    #For file in band in day...
    for date in new_dates:
        phot = pp.read_photometry(date,'Monitoring',targname = targname)
        if phot == None:
            "Problem with " + date + " with " + targname + "! There seems to be no photometry!"
        for band in phot.keys():
            fluxes = {}
            for star in names:
                fluxes[star] = []
            bjd = []
            airmass = []
            try:
                for file in range(len(phot[band]['filename'])):
                    if len(set(flags) & set(phot[band][targname][file]['flags'])) == 0:
                        bjd.append(phot[band]['bjd'][file])
                        airmass.append(phot[band]['airmass'][file])
                        for star in names:
                            fluxes[star].append(phot[band][star][file]['flux'][2])
                #Add the dictionaries to the end of all_bands
                if len(bjd) != 0:
                    for star in names:
                        all_bands[band][star].append(np.array(fluxes[star]))
                    all_bands[band]['bjd'].append(np.array(bjd))
                    all_bands[band]['airmass'].append(np.array(airmass))
            except:
                print "Problem with " + band + " on " + date + " for " + targname

    filename = paths['output']+targname+'_parsed_photometry.pck'
    print "Writing to file " + filename
    outfile = open(filename,'wb')
    pickle.dump(all_bands,outfile)
    outfile.close()
    if output:
        return all_bands
    else:
        return

def parse_photometry(obstype='Monitoring',targname=None,flag = ['saturation','snr','airmass','fwhm','temperature'], write = True,
                     output = False,clobber=True, night_average=True,
                     badrefs=None):
    """
    Description: Updates 'targname_parsed_photometry.pck'
    with all data in /obstype/targname. Automatically finds
    new dates to add.
    -------------
    Inputs: flag = String array; array of flags to be used in parsing photometry.
                   Supported flags: ['saturation','shape','snr','airmass','fwhm','temperature']
            write = bool; writes to targname_parsed_photometry.pck with the latest parsed photometry
            rotated_flat = bool; wether or not to use rotated_flats
            output = bool; print result to screen (do not reccomend, very long)
            badrefs = integers; the number of the reference stars that are bad
    -------------
    Output(s): all_apertures
    -------------
    Example: parse_photometry()
    """
    # Setting up...
    targ_info = pp.read_targ_info(obstype=obstype,targname=targname)
    names = [targname]
    for i in range(len(targ_info['ap_max'])-1):
        names.append('ref'+str(i+1))

    if badrefs is not None:
        bad = ['ref'+str(i) for i in badrefs]
    else:
        bad = []

    names = [val for val in names if val not in bad]

    fmt = '%Y%m%d'
    paths = pp.get_paths(obstype=obstype,targname=targname)

    dirfiles = os.listdir(paths['output'])
    phot_dates = [int(dirfiles[i]) for i in range(len(dirfiles)) if dirfiles[i][0:1] == '2']

    if night_average:
        outname = targname+'_parsed_photometry_nightave.pck'
    else:
        outname = targname+'_parsed_photometry.pck'
    filename = paths['output']+outname

    if len(glob.glob(filename)) == 1:
        if clobber:
            os.system('rm '+filename)
            new_dates = phot_dates
            new_dates.sort()
            all_apertures=[{},{},{},{}]
        else:
            all_apertures = pickle.load(open(filename,'rb'))
            parsed_end_date = find_parsed_end_date()
            print "Latest parsed date", parsed_end_date

            #Check
            if int(parsed_end_date) == max(phot_dates):
                print(targname+"_parsed_photometry is already up-to-date with completed photometry!")
                return all_apertures
            else:
                #Find the dates that have had photometry performed but have not been parsed &
                # added to KIC8462852_parsed_photometry
                new_dates = [phot_dates[i] for i in range(len(phot_dates)) if phot_dates[i] > int(parsed_end_date)]
                new_dates.sort()
    else:
        new_dates = phot_dates
        new_dates.sort()
        all_apertures=[{},{},{},{}]

    #Reject any manually ommited: NEEDS WORK!!!
    #no_dates = obs['date'][obs['use'] == 'N'].values
    no_dates = []
    new_dates = [new_dates[i] for i in range(len(new_dates)) if new_dates[i] not in no_dates]

    #print(new_dates)
    """Iterations: night, band, aperture, sometimes names."""
    #Iterate through each day
    pbar = tqdm(desc = 'Compiling photometry ', total = len(new_dates), unit = 'dates')

    for date in new_dates:
        success = True
        date = str(date)
        phot = pp.read_photometry(date,obstype,targname)
        if phot is None:
            print 'No data found for' + date +'! Look into this!'
        elif len(phot) == 0:
            print 'No data found for' + date +'! Look into this!'
        else:
            #Loops: for ap(4), for band (5), check the files & each star
            for ap in range(4):
                #Loop through each aperture.
                for band in phot.keys():
                    if band not in all_apertures[ap].keys():
                        all_apertures[ap][band]= {}
                    #Set up empty dictionaries for fluxes and bjd and airmass!
                    fluxes = {}
                    for star in names:
                        fluxes[star] = []
                    bjd = []
                    airmass = []
                    try:
                        bjd_all = []; airmass_tally = []
                        # Loop through each file in the band for the day and make
                        # averages
                        for nfile in range(len(phot[band]['filename'])):
                            bad = False
                            for name in names:
                                flagval = phot[band][name][nfile]['flags']
                                for val in flag:
                                    if val in flagval:
                                        bad = True
                            if not bad:
                                bjd_all.append(phot[band]['bjd'][nfile])
                                airmass_tally.append(phot[band]['airmass'][nfile])
                                for star in names:
                                    starflux = phot[band][star][nfile]['flux'][ap]
                                    fluxes[star].append(starflux)

                        if not night_average:
                            for star in names:
                                if star not in all_apertures[ap][band].keys():
                                    all_apertures[ap][band][star] = {'flux':[]}

                                all_apertures[ap][band][star]['flux'] = \
                                    np.append(all_apertures[ap][band][star]['flux'],\
                                              np.array(fluxes[star]))

                            if 'bjd' not in all_apertures[ap][band].keys():
                                all_apertures[ap][band]['bjd'] = []
                            if 'airmass' not in all_apertures[ap][band].keys():
                                all_apertures[ap][band]['airmass'] = []
                            all_apertures[ap][band]['bjd'] = \
                                np.append(all_apertures[ap][band]['bjd'],np.array(bjd_all))
                            all_apertures[ap][band]['airmass'] = \
                                np.append(all_apertures[ap][band]['airmass'],np.array(airmass_tally))
                        else:
                            if len(bjd_all) > 1:
                                star = targname
                                targ = fluxes[star]
                                refsum = []
                                reftally = np.zeros(len(targ))
                                for name in names[1:]:
                                    reftally += fluxes[name]
                                relflux = np.mean(targ/reftally)
                                relerr = np.std(targ/reftally,ddof=1)

                                if star not in all_apertures[ap][band].keys():
                                    all_apertures[ap][band][star] = {'relflux':[],'relerr':[]}

                                all_apertures[ap][band][star]['relflux'] = \
                                    np.append(all_apertures[ap][band][star]['relflux'],\
                                              np.array(relflux))

                                all_apertures[ap][band][star]['relerr'] = \
                                    np.append(all_apertures[ap][band][star]['relerr'],\
                                              np.array(relerr))

                                if 'bjd' not in all_apertures[ap][band].keys():
                                    all_apertures[ap][band]['bjd'] = []
                                if 'airmass' not in all_apertures[ap][band].keys():
                                    all_apertures[ap][band]['airmass'] = []
                                all_apertures[ap][band]['bjd'] = \
                                    np.append(all_apertures[ap][band]['bjd'],np.mean(bjd_all))
                                all_apertures[ap][band]['airmass'] = \
                                    np.append(all_apertures[ap][band]['airmass'],np.mean(airmass_tally))
                            else:
                                pass

                    except:
                        print "Problem with "+date +" in "+band+" band."
                        success = False

        if success:
            print "Added "+date+" to parsed photometry."

        pbar.update(1)
    pbar.close()

    #Write to targname_parsed_photometry.pck
    if write:
        print "Writing to file " + filename
        outfile = open(filename,'wb')
        pickle.dump(all_apertures,outfile)
        outfile.close()
    if output:
        return all_apertures
    else:
        return None

def lightcurve(targname = None, bands = None, startbjd = 2458500., save = True, csv = True,
               utc = False, trend = [2458500,2458500], year = None, error_cap = 1.72e-3):
    #Setting up
    paths,names,refs,targ_info,obs,obs_dates = setup(targname)
    plt.ion()
    fig = plt.figure(figsize = (12,5))
    plt.clf()
    ax = fig.add_subplot(111)
    all_phot = pickle.load(open(paths['output']+targname+'_parsed_photometry_nightave.pck','rb'))
    colors = {'V':'dodgerblue','g':'green','r':'tomato','i':'maroon','z':'grey'}
    band_labels = {'V':'V','g':r'$g^\prime$','r':r'$r^\prime$','i':r'$i^\prime$','z':r'$z^\prime$'}

    if bands == None:
        bands = all_phot[2].keys()
    for band in bands:
        for night in all_phot[2][band]['bjd']:
            night_bjd = np.mean(night)
        #BUGGY startbjd
        relbjd = night_bjd-startbjd
        ut = Time(night_bjd,format='jd',scale='utc',out_subfmt='date_hms')


        ratio = all_phot[2][band][targname]['relflux']
        #Avg ratio of night
        night_ratio = np.array([np.mean(night) for night in ratio])
        #SDOM for ratio of night (Error)
        yerr = all_phot[2][band][targname]['relerr']

        #Filter SDOM of over error_cap (final flag)
        ut.plot_date = ut.plot_date[[yerr.index(i) for i in yerr if i < error_cap]]
        relbjd = relbjd[[yerr.index(i) for i in yerr if i < error_cap]]
        night_ratio = night_ratio[[yerr.index(i) for i in yerr if i < error_cap]]
        yerr = [i for i in yerr if i < error_cap]
        night_bjd = night_bjd[[yerr.index(i) for i in yerr if i < error_cap]]

        #Normalization for now is simply the median ratio in night_ratio
        norm = np.median(night_ratio)

        #CSVs for possible music making at any point becuase litereally why not
        data = {band: band, 'bjd':night_bjd, 'relflux':night_ratio/norm}
        csvdata = pd.DataFrame(data)
        csvdata.to_csv(paths['output']+targname+'_data.csv', index = False)

        if trend != [2458500,2458500]:
            band_labels = {'V':'','g':'','r':'','i':''}
            x = relbjd[ trend[0]-startbjd : trend[1]-startbjd ]
            y = norm_array/norm
            fit_line(x,y,yerr=np.array(yerr[date_range[band][0]:date_range[band][1]])/norm,band = band)

        #Changing the x axis for utc or bjd
        if utc:
            plt.errorbar(ut.plot_date,night_ratio/norm,yerr=np.array(yerr)/norm, markersize = 2.5,fmt = 'o',label = band_labels[band],color=colors[band])
            ax.xaxis_date()
            fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
            date_formatter = mdates.DateFormatter('%m/%d/%Y')
            days = mdates.DayLocator()
            ax.xaxis.set_major_formatter(date_formatter)
            ax.xaxis.set_minor_locator(days)
            ax.set_xlabel('UTC Date')

        else:
            plt.errorbar(relbjd,night_ratio/norm,yerr=np.array(yerr)/norm, markersize = 2.5,fmt = 'o',label = band_labels[band],color=colors[band])
            ax.set_xlabel('JD-2458500')
            if year == None:
                plt.xlim(min(relbjd)-15,max(relbjd)+15)
                plt.xticks(np.arange(int(min(relbjd))-int(min(relbjd)%10)-15,int(max(relbjd))+int(min(relbjd))%10+15, step = 5))
            else:
                plt.xlim(-46+365*(year-2017),-46+365*(year-2016))
                plt.xticks(np.arange(-45+365*(year-2017),-45+365*(year-2016),step = 15))

    out = {'median_bjd':relbjd,'normalized_flux':night_ratio/norm,'error':np.array(yerr)/norm}
    df = pd.DataFrame.from_dict(out)
    df.to_csv(paths['output']+'plots/updates/Update_'+band+'.csv', index_label = False)

    #Plot prettying
    plt.axhline(1.0,linestyle = '--')
    #plt.ylim(0.95,1.02)
    ax.set_ylabel('Relative Flux')
    plt.legend(loc = 'lower right')
    plt.title('Thacher Monitoring of ' + targname)

    #Save with proper suffix
    if save:
        tag=''
        if len(bands)==1:
            tag+='_'+band
        if bands == all_phot.keys() and len(bands) != 1:
            tag += '_all'
        if year != None:
            tag+='_'+str(year)
        if utc:
            tag+='_utc'
        outfile = paths['output']+'plots/lightcurves/'+targname+'_'+tag+'.png'
        print "Saving in " + outfile
        plt.savefig(outfile,dpi=300)

    return


def lightcurve_devel(targname = None, bjd_ref = 2458500., save = True, csv = True,
               utc = False, trend = [2458500,2458500], bands = ['g','r','i','z'], year = 2019, error_cap = 0.005,markersize=5,
               zoom=True,zoom_start=290,zoom_tick=5,tag='', night_average=True, ydepth=0.8, ylimit=1.5, xlim=370):
    #Setting up
    paths,names,refs,targ_info,obs,obs_dates = setup(targname)
    plt.ion()
    fig = plt.figure(figsize = (12,5))
    plt.clf()
    ax = fig.add_subplot(111)
    if night_average:
        all_phot = pickle.load(open(paths['output']+targname+'_parsed_photometry_nightave.pck','rb'))
    else:
        all_phot = pickle.load(open(paths['output']+targname+'_parsed_photometry.pck','rb'))
    colors = {'V':'dodgerblue','g':'green','r':'tomato','i':'maroon','z':'grey'}
    band_labels = {'V':'V','g':r'$g^\prime$','r':r'$r^\prime$','i':r'$i^\prime$','z':r'$z^\prime$'}
    #Need to set range ERROR HERE norm_array reading nans
    #NSVS8923588:
    date_range = {'g':[1,8],'r':[1,8],'i':[1,8],'z':[1,8]}

    #date_range = {'g':[0,1],'r':[0,1],'i':[0,1],'z':[0,1]}

    if bands == None:
        bands = all_phot[2].keys()
    if not night_average:
        date_range = {'g':[5,8],'r':[5,8],'i':[5,8],'z':[5,8]}
        for band in bands:
            bjd = all_phot[2][band]['bjd']
            ut = Time(bjd,format='jd',scale='tdb',out_subfmt='date_hms')

            flux = all_phot[2][band][targname]['flux']

            #norm_array = flux[date_range[band][0]:date_range[band][1]]
            # Sigma rejected mean??
            #norm = np.mean(norm_array)

            #normflux = flux/norm
            #Relative bjd
            rel_bjd = bjd-bjd_ref
            plt.scatter(rel_bjd,flux,marker='o',label=band_labels[band], color=colors[band])
            ax.set_xlabel('JD-2458500',fontsize=13)
            if year == None:
                add = np.round((max(rel_bjd)-45)*0.05)
                plt.xlim(45,max(rel_bjd)+add)
                plt.xticks(np.arange(50,max(rel_bjd)+add, step = 50))
            else:
                plt.xlim(-46+365*(year-2017),-46+365*(year-2016))
                plt.xticks(np.arange(-45+365*(year-2017),-45+365*(year-2016),step = 20))
                if zoom:
                    #plt.xlim(365*(year-2017),max(rel_bjd)+3)
                    add = np.round((max(rel_bjd)-zoom_start)*0.05)
                    plt.xlim(zoom_start,max(rel_bjd)+add)
                    plt.xticks(np.arange(zoom_start,max(rel_bjd)+add, step = zoom_tick))

    else:
        for band in bands:
            bjd = all_phot[2][band]['bjd']
            ut = Time(bjd,format='jd',scale='tdb',out_subfmt='date_hms')

            relflux = all_phot[2][band][targname]['relflux']
            relflux_err = all_phot[2][band][targname]['relerr']

            norm_array = relflux[date_range[band][0]:date_range[band][1]]
            # Sigma rejected mean??
            norm = np.mean(norm_array)

            normflux = relflux/norm
            normerr = relflux_err/norm

            # Reject data above error threshold
            good, = np.where(normerr < error_cap)

            #Relative bjd
            rel_bjd = bjd-bjd_ref

        # #Get slope of data points
        # if trend:
        #     band_labels = {'V':'','g':'','r':'','i':''}
        #     x = rel_bjd[date_range[band][0]:date_range[band][1]]
        #     y = norm_array/norm
        #     fit_line(x,y,yerr=np.array(yerr[date_range[band][0]:date_range[band][1]])/norm,band = band)

        #Changing the x axis for utc or bjd
            if utc:
                plt.errorbar(ut.plot_date[good],normflux[good],yerr=normerr[good],
                             markersize=markersize,fmt='o',label=band_labels[band],
                             color=colors[band])

                ax.xaxis_date()
                fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
                date_formatter = mdates.DateFormatter('%m/%d/%Y')
                days = mdates.DayLocator()
                ax.xaxis.set_major_formatter(date_formatter)
                ax.xaxis.set_minor_locator(days)
                ax.set_xlabel('UTC Date')

            else:
                plt.errorbar(rel_bjd[good],normflux[good],yerr=normerr[good],
                             markersize=markersize,fmt='o',label=band_labels[band], color=colors[band])
                ax.set_xlabel('JD-2458500',fontsize=13)
                if year == None:
                    add = np.round((max(rel_bjd)-45)*0.05)
                    plt.xlim(45,max(rel_bjd)+add)
                    plt.xticks(np.arange(50,max(rel_bjd)+add, step = 50))
                else:
                    plt.xlim(-46+365*(year-2017),-46+365*(year-2016))
                    plt.xticks(np.arange(-45+365*(year-2017),-45+365*(year-2016),step = 20))
                    if zoom:
                        #plt.xlim(365*(year-2017),max(rel_bjd)+3)
                        add = np.round((max(rel_bjd)-zoom_start)*0.05)
                        plt.xlim(zoom_start,max(rel_bjd)+add)
                        plt.xticks(np.arange(zoom_start,max(rel_bjd)+add, step = zoom_tick))

            #Add csv file in each band in output directory
            if csv:
                out = {'median_bjd':rel_bjd[good],'normalized_flux':normflux[good],'error':np.array(normerr[good])/norm}
                df = pd.DataFrame.from_dict(out)
                df.to_csv(paths['output']+targname+'_update_'+band+'.csv')

    #Plot prettying
    if not night_average:
        #plt.axhline(1.0,linestyle = '--')
        plt.ylim(3000,8000)
        #plt.xlim(295,318)
        ax.set_ylabel('Flux',fontsize=13)
        plt.legend(loc = 'best')
        plt.title('Thacher Nightly Monitoring of '+targname,fontsize=15)
    if night_average:
        plt.axhline(1.0,linestyle = '--')
        plt.ylim(ydepth,ylimit)
        #plt.xlim(295,xlim)
        ax.set_ylabel('Relative Flux',fontsize=13)
        plt.legend(loc = 'best')
        plt.title('Thacher Monitoring of '+targname,fontsize=15)

    #Save with proper suffix
    if save:
        outfile = paths['output']+'plots/lightcurves/Lightcurve'+tag+'.png'
        print "Saving in " + outfile
        plt.savefig(outfile,dpi=300)
    return

"""
Used/Useful functions below
"""

def daily(targname):
    paths,names,refs,targ_info,obs,obs_dates = setup(targname)
    do_phot(targname)
    parse_photometry(targname)
    lightcurve(targname)

def NSVS_daily(targname=None):
    paths,names,refs,targ_info,obs,obs_dates = setup(targname)
    do_NSVS_phot(targname=targname)
    parse_photometry(targname=targname)
    lightcurve_devel(targname=targname)

# mp.daily('Mrk501')
# mp.daily('PKS1510-089')


def do_NSVS_phot(obstype='Monitoring', targname=None, rotated_flat = True):
    """
    Description
    -----------
    Performs photometry for any new nights found in Archive that have NSVS Monitoring data!
    Also updates targname_summary file
    -------------
    Inputs: None
    -------------
    Output(s): Creates a new directory for any new days and adds:
               - Master Bias .png and .fits
               - Master Dark .png and .fits
               - photometry_(date).pck
    -------------
    Example: do_NSVS_phot()
    """
    #-----Update obs_summary.csv with any data from Archive------#

    update = pp.update_summary_file(obstype=obstype,targname=targname)

    #Get paths, and dates
    paths = pp.get_paths(targname=targname,obstype=obstype)
    obs = pp.get_summary(obstype,targname)
    obs_dates = obs['date'][obs['use'] == 'Y'].values
    dirfiles = os.listdir(paths['output'])

    phot_dates = [int(dirfiles[i]) for i in range(len(dirfiles)) if dirfiles[i][0:1] == '2']
    phot_dates = [phot_dates[i] for i in range(len(phot_dates)) if paths['output']+str(phot_dates[i])+
                  '/'+"photometry_"+str(phot_dates[i])+'.pck' is not None]
    phot_dates.sort()

    #Find new dates in summary file that have not yet done photometry.
    new_dates = [str(obs_dates[i]) for i in range(len(obs_dates)) if obs_dates[i] not in phot_dates]

    if len(new_dates) == 0:
        print "No new photometry to be done since ", phot_dates[-1]
    else:
        print "Performing photometry on the following days"
        print new_dates

    #Do night_phot for each new day.
    for date in new_dates:
        pp.night_phot(date,obstype=obstype,targname=targname,rotated_flat=rotated_flat,
                      outdir=paths['output'], calwrite=False, calarchive=True)

    print "Finished photometry for all new dates"
