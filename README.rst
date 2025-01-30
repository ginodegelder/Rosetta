=============
Rosetta
=============
-----------------------------------------------
Reconstruction Of Sea-level, Ecosystems and Tectonics from Terrace Analysis
-----------------------------------------------

Description
===========

| This model is a combination of a coral reef construction model (new version of REEF; Husson et al., 2018; Pastier et al., 2019) and a bayesian inversion model (Hedjazian et al., 2019). The resulting ROSETTA model runs a given number of forward REEF models, in order to diminish the misfit between the modelled and observed 1-D topographic profile, using a Monte-Carlo Markov Chain (MCMC) sampling. Based on a Bayesian framework, the solution of our model is a posterior probability distribution describing the probability of the model parameters (here the past sea-level variations and the REEF parameters), given the observed data (here the topo-bathymetry of a marine terrace sequence).
| Here is given the full package containing a switch to an erosive only mode, and using the last version of the forward REEF model. If you want to reproduce the outputs from de Gelder et al. (Submitted), download the version and inputs in the "erosive_only" branch of this repository.
|
| Further reading:

de Gelder, G., Hedjazian, N., Husson, L., Bodin, T., Pastier, A.-M., Boucharat, Y., Pedoja, K., Solihuddin, T. and Cahyarini, S.Y. (Submitted) Bayesian reconstruction of sea-level and hydroclimates from coastal landform inversion, submitted to Earth Surface Processes and Landforms. Preprint: https://doi.org/10.31223/X5B117

Hedjazian, N., Bodin, T., & Métivier, L. (2019). An optimal transport approach to linearized inversion of receiver functions. Geophysical Journal International, 216(1), 130-147.

Husson, L., Pastier, A. M., Pedoja, K., Elliot, M., Paillard, D., Authemayou, C., ... & Cahyarini, S. Y. (2018). Reef carbonate productivity during quaternary sea level oscillations. Geochemistry, Geophysics, Geosystems, 19(4), 1148-1164.

Pastier, A. M., Husson, L., Pedoja, K., Bézos, A., Authemayou, C., Arias‐Ruiz, C., & Cahyarini, S. Y. (2019). Genesis and architecture of sequences of Quaternary coral reef terraces: Insights from numerical models. Geochemistry, Geophysics, Geosystems, 20(8), 4248-4272.

Installation 
------------

Environment required : Linux or Mac

RoSETTA strongly relies on MPI (running with mpi4py), make sure you have an executable mpirun command by running::

	which mpirun

Which should give an output of the form : /usr/bin/mpirun
At the moment, Rosetta has been tested using Open MPI 4.1.4. To check your version::

	mpirun --version

Do the same with python, and make sure you are using a python3 version.
If you want to run the model on a cluster/HPC, you might need to create a virtual environment to install all the libraries needed for the model. You can skip this part if you run the model locally.
Creating a python virtual environment named 'virtualenv' in 'home' directory::

        yboucharat@Yannick:~$ python3.11 -m venv virtualenv

Activation::

        yboucharat@Yannick:~$ source virtualenv/bin/activate

You are now in your python3.11 virtual environment where you can install the needed libraries.

To install the needed libraries::

        pip install -r requirements.txt

Go to Library/reef and run in the terminal::

	cd Library/reef
	pythran PythranTools.py

This will create a .so file which will help for optimisation.
To quit the virtual env::
        
        deactivate

Initialisation :
----------------

You should only modify "Inputs.py" and add you observed topographic profiles in "./Topo_obs/" if you just want to run the model. 

To access source code of the reef construction, everything is in the folder "Library/reef". 
To access source code of Bayesian MCMC inversion, it's in "Library/mcmc".

Everything is detailed in "Inputs.py". Just follow the indicated format for the dictionnaries.

Put every observed topography files, in a .dat format, in the Topo_obs directory. If you want to use another directory or path, don't forget to change the path in "Inputs.py".
The files with the observed topographies have to be of the following form (x and y separated by tab) :

   - x(m): Distance in meters
   - y(m): Elevation in meters
   - Do not write the header, only topo values

+------+------+
| x(m) | y(m) |
+======+======+
| 0    | -40  |
+------+------+
| ...  | ...  |
+------+------+
| 1000 | 0    |
+------+------+


Run the model :
---------------

 - If you can use one core for each profile, you can run the model as follows::

        mpirun -np <number of cores/profiles> --machinefile "machinefile.txt" python3 Rosetta.py

 - If the number of cores and number of topographic profiles are different, you need to add the "-oversubscribe" argument::

        mpirun -np <number of topo profiles> -oversubscribe --machinefile "machinefile.txt" python3 Rosetta.py

--machinefile can be replaced by --hostfile, they are synonyms. "machinefile.txt" is a simple text file with the IP adresses of the computing cores you are using. For more details : https://docs.open-mpi.org/en/v5.0.x/man-openmpi/man1/mpirun.1.html#label-schizo-ompi-hostfile

The time needed for the inversion varies a lot with the time length of one simulation (older value in sea_level) but also with a wide initial platform ("init__lterr") and gentle initial slope ("grid__slopi"). Try with approximately 10 to a 1000 simulations (n_samples) to see how much time is needed for 1 simulation on average.
If you run the model in passive mode on a HPC, you can display the output (assuming the output file is called "output.stdout") with::

	tail -f output.stdout

The first iteration is approximately 2 to 3 times slower than the next ones, wait some iterations to have a better idea of the computation time. 

Outputs :
---------

All the outputs are in "Outs/FigS4d/" folder.
The outputs from one model are stored in a directory named : "Figs_<n_samples>_sig.<sigma>_ip.<ipstep>_<DAY-MONTH-YEAR_HOUR-MIN at the end of the run>".
 - There will be one subfolder for each profile, named by the name of the profile, containing the histogramms for each reef parameters corresponding to the profile, its best, mean and median profile in .txt and the profile figures. 
 - "Dataframes" folder contains all the data saved at the end of the inversion. Data are saved in .pkl format, readable with panda.
 - "SL" folder contains histogramms of the free SL nodes, best, mean and median SL history in .txt format, and SL plots.
 - "Stats" folder contains the evolution of the best acceptance ratio of the profiles, the best loglikelihood, the acceptance ratio for each parameters, and the step evolution (prop_S) for each free parameters. 
 - "AA-Inputs" file contains all the Inputs for the model.
 - "BestLogLike" contains the value of the best loglike reached during the inversion.

Example of an output folder in "Outs/FigS4d"

::

    Figs_10_sig.10_ip.150_19-02-2024_17.01
    ├── Dataframes
    │    ├── MCMC_raw.nc            # Raw output in netcdf file. Can be read with arviz library on python
    │    ├── df_SL_6.0ky.pkl
    │    ├── My_topo_file_1_param1-param2.pkl
    │    ├── [other files for topo_file_1]
    │    ├── My_topo_file_2_param1-param2.pkl
    │    └── [other files for topo_file_2]
    ├── SL
    │    ├── BestSL.txt
    │    ├── Histogram-6.0ky.png
    │    ├── MeanSL.txt
    │    ├── MedianSL.txt
    │    ├── Sea-Level.pdf
    │    └── Sea-Level_median_percentiles.pdf
    ├── Stats
    │    ├── Accept_ratio.png
    │    ├── Loglikelihood.png
    │    ├── Param_accept_ratio.png
    │    └── Prop_S.png
    ├── My_topo_file_1
    │    ├── BestProfile.txt
    │    ├── [2D Histograms for topo 1 free reef params]
    │    ├── MeanProfile.txt
    │    ├── MedianProfile.txt
    │    ├── Profile_median_percentiles.pdf
    │    └── Profiles.pdf
    ├── My_topo_file_2
    │    ├── BestProfile.txt
    │    ├── [2D Histograms for topo 1 free reef params]
    │    ├── MeanProfile.txt
    │    ├── MedianProfile.txt
    │    ├── Profile_median_percentiles.pdf
    │    └── Profiles.pdf
