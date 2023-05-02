# Data-Driven Guided Attention for Analysis of Physiological Waveforms with Deep Learning

This respository hosts the research for the manuscript entitled, "Data-Driven Guided Attention for Analysis of Physiological Waveforms with Deep Learning". The repo is organized as follows:

Data Annotation - Hosts the files necessary for annotating the fiducial points for both the Bio-Z and MIMIC PPG modalities with Boosted-SpringDTW.
DDGA - Hosts the files necessary for running DDGA for the Personalized experiments and the Interpolation and Extrapolation experiments.
DDGA/Personalized/ - These experiments train on a given subjects data and test on the same subjects data.
DDGA/Interploation and Extrapolation/ - These experiments train on specific ranges of a person's BP data and tests on the remaining BP ranges.
