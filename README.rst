=============
RoSETTA erosive only version
=============
-----------------------------------------------
Reconstruction of SEa-level and Tectonics from Terrace Analysis
-----------------------------------------------

Description
===========

| This model is a combination of a coral reef construction model (new version of REEF; Husson et al., 2018; Pastier et al., 2019) and a bayesian inversion model (Hedjazian et al., 2019). The resulting ROSETTA model runs a given number of forward REEF models, in order to diminish the misfit between the modelled and observed 1-D topographic profile, using a Monte-Carlo Markov Chain (MCMC) sampling. Based on a Bayesian framework, the solution of our model is a posterior probability distribution describing the probability of the model parameters (here the past sea-level variations and the REEF parameters), given the observed data (here the topo-bathymetry of a marine terrace sequence).
| Here is given an erosive only version of RoSETTA, with all scripts and inputs used in deGelder et al. (submitted). This version has not yet been converted to the more user-friendly version of RoSETTA, but will be updated soon. The version of RoSETTA given in the main branch of this repository gives the possibility to run it in an erosive only mode.
|
| Further reading:

de Gelder, G., Hedjazian, N., Husson, L., Bodin, T., Pastier, A.-M., Boucharat, Y., Pedoja, K., Solihuddin, T. and Cahyarini, S.Y. (Submitted) Bayesian reconstruction of sea-level and hydroclimates from coastal landform inversion, submitted to Earth Surface Processes and Landforms. Preprint: https://doi.org/10.31223/X5B117

Hedjazian, N., Bodin, T., & Métivier, L. (2019). An optimal transport approach to linearized inversion of receiver functions. Geophysical Journal International, 216(1), 130-147.

Husson, L., Pastier, A. M., Pedoja, K., Elliot, M., Paillard, D., Authemayou, C., ... & Cahyarini, S. Y. (2018). Reef carbonate productivity during quaternary sea level oscillations. Geochemistry, Geophysics, Geosystems, 19(4), 1148-1164.

Pastier, A. M., Husson, L., Pedoja, K., Bézos, A., Authemayou, C., Arias‐Ruiz, C., & Cahyarini, S. Y. (2019). Genesis and architecture of sequences of Quaternary coral reef terraces: Insights from numerical models. Geochemistry, Geophysics, Geosystems, 20(8), 4248-4272.

Installation 
------------

Environment required : Linux or Mac
At the moment, this version of RoSETTA is relying only on python.
Make sure you are using a python3 version::

        python3 --version

Which should give an output of the form (depending on your version) : Python 3.12.3

If you want to run the model on a cluster/HPC, you might need to create a virtual environment to install all the libraries needed for the model. You can skip this part if you run the model locally.
Creating a python virtual environment named 'virtualenv' in 'home' directory::

        cd ~/
	python3 -m venv virtualenv

Activation::

        source virtualenv/bin/activate

You are now in your python3.11 virtual environment where you can install the needed libraries.

Now, git clone this repository::

	git clone https://github.com/ginodegelder/Rosetta

This will clone the whole repository, to extract this version, run::

        cd Rosetta
        git checkout erosive_only

To install the needed libraries (activate your virtual environment if needed)::

        pip install -r requirements.txt

To quit the virtual env::
        
        deactivate

Run the model :
----------------

As this version is not really user-friendly and will be updated soon, we will not explain here all the details of the source code used here, and how to make your own setup. For this, use the version given in the main branch of this repository.

To reproduce the figures from deGelder et al. (submitted), you will need to run the scripts inside the "Scripts_deGelder_et_al_2025" folder.::

        cd Scripts_deGelder_et_al_2025
        python3 ManuscriptFig<figure number>

The computing time, from one script to another, is from several hours to approximately 2 days. You might want first to test the model and see if it's working. For shorter runs, open the script you want to test, and modify the number of samples (ie.the number of forward models).::

        line 23 : n_samples = 1000000

You can try with 10 to 1000 samples, before running the full model.

This will save the figures in the "Figs/Fig<figure number>/" folder.

