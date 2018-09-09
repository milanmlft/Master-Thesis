# Modeling Tumor Heterogeneity 
### Milan Malfait
### ULB, Interuniversity Institute for Bioinformatics in Brussels (IB<sup>2</sup>)
Master thesis submitted to obtain the degree of Master in Bioinformatics and Modeling at the Université Libre de Bruxelles (Brussels, Belgium).

# Table of contents for GitHub repository

## 1. Dissertation
[PDF version](https://github.com/milanmlft/Master-Thesis/blob/master/Thesis_Milan_Malfait_FINAL.pdf)


## 2. Model and scripts
The model is described (apart from the Methods section in the dissertation) in the [Model](https://github.com/milanmlft/Master-Thesis/blob/master/Model.ipynb) notebook.

Python scripts that implement the model and used to simulate, analyze and visualize tumor populations are contained within the [ThesisScripts](https://github.com/milanmlft/Master-Thesis/tree/master/ThesisScripts) module.


## 3. Simulations
Jupyter Notebooks containing the code to simulate the tumor populations.

### 3.1 [Individual simulations](https://github.com/milanmlft/Master-Thesis/tree/master/Simulations/Individual-simulations)
Simulations and analyses of individual tumor populations under different parameter sets. Figures from the resulting analyses are stored in a separate directory.

### 3.2 [Ensemble simulations](https://github.com/milanmlft/Master-Thesis/tree/master/Simulations/Ensemble-simulations)
To get statistical significant results, simulations of a tumor population with a given parameter set were repeated 1000 times. The resulting populations of these simulations are stored in the [saved_simulatons](https://github.com/milanmlft/Master-Thesis/tree/master/Simulations/Ensemble-simulations/saved_simulations) directory as pickled gzip files for later use in the analyses.

The notebooks containing **Large** in their filename contain simulations of populations with size **10<sup>8</sup>**.

The notebooks containing **Small** in their filename contain simulations of populations with size **10<sup>4</sup>**.

**WARNING:** the simulations of the *Large* populations can use up a lot of memory and may take 2-3 hours to run.


## 4. Analyses
Jupyter Notebooks used to analyze the ensemble simulations.

### 4.1 [Ensemble analyses](https://github.com/milanmlft/Master-Thesis/tree/master/Analyses/Ensemble-analyses)
Analyses of the ensemble simulations. **Uses the simulation data stored in the [saved_simulatons](https://github.com/milanmlft/Master-Thesis/tree/master/Simulations/Ensemble-simulations/saved_simulations) directory!** Analyzed data is stored in the [Analysis-Data](https://github.com/milanmlft/Master-Thesis/tree/master/Analyses/Analysis-Data) directory.

### 4.2 Comparative analyses
The influence of the level of selective pressure on the tumor populations was assessed by performing comparative analyses, both for **[large](https://github.com/milanmlft/Master-Thesis/blob/master/Analyses/Comparative_analysis-Large.ipynb)** and **[small](https://github.com/milanmlft/Master-Thesis/blob/master/Analyses/Comparative_analysis-Small.ipynb)** populations.

These notebooks use the previously analyzed data stored in the [Analysis-Data](https://github.com/milanmlft/Master-Thesis/tree/master/Analyses/Analysis-Data) directory (instead of repeating the analyses).

### 4.3 [Sampling method analysis](https://github.com/milanmlft/Master-Thesis/blob/master/Analyses/Sampling_method-analysis.ipynb)
Notebook to analyze the influence of the artificial sampling method.

### 4.4 [Selection weights analysis](https://github.com/milanmlft/Master-Thesis/blob/master/Analyses/Selection_weights-analysis.ipynb)
Notebook to visualize the fitness weights distributions for populations grown under various levels of selection. 
