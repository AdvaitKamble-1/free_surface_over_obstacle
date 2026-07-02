# masters_thesis_predictive_modelling

The purpose of this repository is to present the various post-processing Python scripts used in completing this project. It should be noted that these scripts are only applicable to the output binary files of the FDM solver as well as the resultant text files.

The structure of this repository is as follows:
1. Validation Scripts:

   a. bouscasse.py -> Corresponds to comparison of solver results with study by Bouscasse et al. (SPH Modelling of viscous flow past a circular cylinder interactin with a free-surface)

   b. reichl.py -> Corresponds to comparison of solver results with study by Reichl et al. (Flow past a cylinder close to a free-surface)
3. Experimental Processing (Note: this is not processing experimental data, only extracting useful information from the binary obtained using the solver)

   a. amplitude_contour.py -> Used to extract interface deformation data from the output binary files.

   b. Plotting.py -> Used to reshape binary data (u*.bin, v*.bin, vof*.bin) into numpy arrays and plot velocity contours.

   c. wave_tracking.py -> Used to identify wave-tracks, construct a linear fit, overlay on the amplitude contour, and compare wave-speeds against those obtained via the linear dispersion relation.

   d. grid_resolution.py -> Used to compare various grid resolutions (coarse, medium, fine) and their effect on the flow development,

Note: use of AI (Claude) was primarily for assisting in writing loops, generating skeleton scripts and fine-tuning of parameters (threshold selection, etc.)
