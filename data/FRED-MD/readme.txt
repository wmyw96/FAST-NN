readme.txt
===========================================================================
DESCRIPTION
This zip file contains MATLAB code for processing the FRED-MD dataset and 
then estimating factors. Monthly and quarterly versions of the dataset can 
be found at http://research.stlouisfed.org/econ/mccracken/fred-databases/. 
The code loads in the data, transforms each series to be stationary, 
removes outliers, estimates factors, and computes the R-squared and 
marginal R-squared values from the estimated factors and factor loadings.

===========================================================================
LIST OF FILES
This zip file contains one main MATLAB script and a set of auxiliary 
MATLAB functions that are called from the main script. The auxiliary 
functions should be saved in the same folder as the main script.

---------------------------------------------------------------------------
MAIN SCRIPT

fredfactors.m     

    This is the main MATLAB script. It performs all the tasks mentioned 
    above using the auxiliary functions described below. 

---------------------------------------------------------------------------
AUXILIARY FUNCTIONS

prepare_missing.m           

    MATLAB function that transforms the raw data into stationary form. 

remove_outliers.m            

    MATLAB function that removes outliers from the data. A data point x is 
    considered an outlier if |x-median|>10*interquartile_range. 

factors_em.m            

    MATLAB function that estimates a set of factors for a given dataset 
    using principal component analysis. The number of factors estimated is 
    determined by an information criterion specified by the user. Missing
    values in the original dataset are handled using an iterative 
    expectation-maximization algorithm. 

mrsq.m

    MATLAB function that computes the R-squared and marginal R-squared 
    values from estimated factors and factor loadings.

===========================================================================


