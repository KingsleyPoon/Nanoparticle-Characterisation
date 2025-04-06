# Introduction
This repository contains the iterative mask refinement (IMR) method , developed in Python, during my doctoral thesis for the automated characterisation of truncated nanocubes segmented from transmission electron micrographs. Although initially designed for only a limited number of similar shapes (_i.e.,_ truncated cubes, spheres, rods), it can be adapted for any mathematically definable shape.

# Code function
The characterisation process is as follows:

1. Reduce image dimensionality (precaution to input of binary RGB images)
2. Identify nanoparticle (NP) regions
3. Remove NP regions based on selection criteria (_e.g.,_ minimum NP area, circularity _etc._)
4. For each individual NP, find the best geometrical model fit.
5. Extract the shape parameters from the geometric model
6. Repeat for all NP regions and collate the shape measurements

# Notes
- The application requires a binary image where NPs have integer values ≥1 and the background is 0. While test images use Labkit segmentation (statistically comparable to manual), other binary segmentation methods can be used for potentially better results. Other segmentation methods in the literature have been provided below for reference.
- This application cannot differentiate between high-quality and low-quality NPs. Therefore, before selecting an image to process, it is important to remove any NPs from the image which are not representative of the sample. This includes:
  * Blurred edges from NP surfactant
  * NPs imaged at an oblique angle (_i.e._ not imaged along the facial plane)
  * Overlapping NPs
  * NPs that are partially within the image frame

# Version Log
**Version 1.**
- Original IMR implementation

**Version 2.**
- Updated interface (application view)
- NP images can now be selected by through a **_Browse_** feature
- Shape parameter measurements can be adjusted by the image scale
- Removed generated figures
- Allow multiple images to be selected and analysed

**Version 3.**
- Parallelisation of NP analysis (speed increase dependent on cpu core count)
- Removed multiple image analysis

# References
- Sun, Z. J.;  Shi, J.;  Wang, J.;  Jiang, M. Q.;  Wang, Z.;  Bai, X. P.; Wang, X. X., A deep learning-based framework for automatic analysis of the nanoparticle morphology in SEM/TEM images. Nanoscale 2022, 14 (30), 10761-10772.
- Wang, X. Z.;  Li, J.;  Ha, H. D.;  Dahl, J. C.;  Ondry, J. C.;  Moreno-Hernandez, I.;  Head-Gordon, T.; Alivisatos, A. P., AutoDetect-mNP: An Unsupervised Machine Learning Algorithm for Automated Analysis of Transmission Electron Microscope Images of Metal Nanoparticles. JACS Au 2021, 1 (3), 316-327.
- Lee, B.;  Yoon, S.;  Lee, J. W.;  Kim, Y.;  Chang, J.;  Yun, J.;  Ro, J. C.;  Lee, J. S.; Lee, J. H., Statistical Characterization of the Morphologies of Nanoparticles through Machine Learning Based Electron Microscopy Image Analysis. ACS Nano 2020, 14 (12), 17125-17133.
- Mondini, S.;  Ferretti, A. M.;  Puglisi, A.; Ponti, A., PEBBLES and PEBBLEJUGGLER: software for accurate, unbiased, and fast measurement and analysis of nanoparticle morphology from transmission electron microscopy (TEM) micrographs. Nanoscale 2012, 4 (17), 5356-5372.
