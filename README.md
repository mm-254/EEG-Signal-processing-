# EEG-Signal-processing-
EEG Signal preprocessing and classification in Matlab.

This code was developed to analyse the ability of EEG signals in discriminating between surface textures silk, suede, and sandpaper during grasp and lift tasks
Data used in this code is data produced by Luciw et.al.(https://www.nature.com/articles/sdata201447#Sec22). A function WEEG_GetEventsInHS() is provided by the authors and used in this code.

Surface series trials of subjects 7 and 11 are used for this project.
This code:
1) Preprocesses the EEG data
2) Performs dimensionality reduction with principle component analysis (this portion is commented out in the code to generate results from subsequent techniques)
3) Generates spectrograms
4) Performs a non parametric permutation test using a function permutest found here: https://www.mathworks.com/matlabcentral/fileexchange/71737-permutest, and 
5) Implements an SVM classifier and computes accuracy, false alarm rate, etc.

Other information:
Mastoid channels are channels 17 and 22.
EEG data is referenced from mastoid channels as signal of interest is in motor cortex. 


The function mttfr() for time-frequency analysis of multiple wavelets was provided in a course with rights reserved. Here is some information provided in the file that might help reproduce it:

- Usage:

  [power, avepow, itc, times] = mttfr(x, fs, freqs, n_cycles, time_bandwidth)

  Example:

  [power, avepow, itc, times] = mtffr(x, 4000, 30:5:100, 7.0, 2.0)

  x : 2-D array of size n_time x n_trials
  Data
  fs : scalar, Hz
  Sampling frequency
  freqs : 1-D vector of size n_freqs x 1
  The set of frequencies to calculate the time-frequency representation
  n_cycles : either a 1-D vector the same size as freqs (or) scalar
  The number of cycles in the wavelet to be used
  time_bandwidth : scalar, unitless should be >= 2. Optional, default 4.0
  Time-(full) bandwith product for the wavelet tapers. The number of
  tapers is automatically chosen based on the time-bandwidth product.
  useparfor : boolean, optional, default is false
  Set to true if parfol is available for your MATLAB version and machine setup to possibly reduce computation time by using parallel threads for different frequencies. Not recommended for small datasets.

  References:

  Slepian, D. (1978). Prolate spheroidal wave functions, Fourier analysis,
  and uncertainty?V: The discrete case. Bell System Technical Journal,
  57(5), 1371-1430.

  Thomson, D. J. (1982). Spectrum estimation and harmonic analysis.
  Proceedings of the IEEE, 70(9), 1055-1096.


