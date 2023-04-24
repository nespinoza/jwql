# Import base libraries:
import glob

# Import classic libraries:
import numpy as np
import matplotlib.pyplot as plt

# Import some JWST useful functions:
from jwst import datamodels

# Import transitspectroscopy:
import transitspectroscopy as ts

def get_soss_traces(image, order1_guess = None, order2_guess = None, xend1 = 4, xend2 = 600):
    
    if order1_guess is None:
        
        xstart1 = 2043
        
        # Perform CCF on Order 1, find maximum, set that as the guess:
        lags1, ccf1 = ts.spectroscopy.get_ccf(np.arange(image.shape[0]), 
                                              image[:, xstart1], 
                                              function = 'double gaussian', 
                                              parameters = ccf_parameters)
        
        idx = np.where(np.max(ccf1) == ccf1)[0]
        ystart1 = lags1[idx][0]
        
        order1_guess = [xstart1, ystart1]
        
    if order2_guess is None:
        
        xstart2 = 1750
        
        # Perform CCF on Order 1, find maximum, set that as the guess:
        lags2, ccf2 = ts.spectroscopy.get_ccf(np.arange(image.shape[0]), 
                                              image[:, xstart2], 
                                              function = 'double gaussian', 
                                              parameters = ccf_parameters)
        
        idx_large = np.where(lags2>200)[0]
        ccf2 = np.array(ccf2)

        idx = np.where( np.max(ccf2[idx_large]) == ccf2[idx_large] )[0]

        ystart2 = lags2[idx_large][idx][0]
        
        order2_guess = [xstart2, ystart2]
        
    x1, y1 = ts.spectroscopy.trace_spectrum(image, np.zeros(image.shape), 
                                           xstart = xstart1, ystart = ystart1, xend = xend1, 
                                           ccf_function = 'double gaussian', 
                                           ccf_parameters = ccf_parameters)
        
    x2, y2 = ts.spectroscopy.trace_spectrum(image, np.zeros(image.shape), 
                                           xstart = xstart2, ystart = ystart2, xend = xend2, 
                                           ccf_function = 'double gaussian', 
                                           ccf_parameters = ccf_parameters)
    
    return x1, y1, x2, y2

def read_dataset(directory, filenames):

class TSOMonitor()
    """Class for executing the TSO monitor.

    """

    def __init__(self):
        """Initialize an instance"""

    def run(self):

        """
        Here we should have a function that looks for all the TSOs in a given 
        time period and runs the monitor below for each TSO *observation*. Below I create 
        a dummy dictionary that has this information put in by hand with a dataset in 
        central store:
        """

        tso_datasets = {}
        tso_datasets['pid_1541_obs1'] = {}
        tso_datasets['pid_1541_obs1']['folder'] = '/ifs/jwst/wit/witserv/data18/nis_commissioning/nis-034_1541/data/01541/obsnum01/vanilla_196/'
        tso_datasets['pid_1541_obs1']['instrument/mode'] = 'niriss/soss'
        tso_datasets['pid_1541_obs1']['files'] = ['jw01541001001_04101_00001-seg001_nis_rateints.fits', \
                                                  'jw01541001001_04101_00001-seg002_nis_rateints.fits', \
                                                  'jw01541001001_04101_00001-seg003_nis_rateints.fits', \
                                                  'jw01541001001_04101_00001-seg004_nis_rateints.fits']
        tso_datasets['pid_1541_obs1']['obs'] = 1

        # We need the instrument/mode because that will define how we preprocess the TSO, and what products we use. We iterate over the observations 
        # and visits and run our algorithms:
        for k in list( tso_datasets.keys() ):

            if tso_datasets[k]['instrument/mode'] in ['niriss/soss', 'nirspec/g395h', 'nirspec/g395m']:

                tso_data = read_dataset(tso_datasets['pid_1541_obs1']['folder'], tso_datasets['pid_1541_obs1']['files'])
