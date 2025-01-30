# SolNet
Open-source tool to get a pre-trained deep learning model for solar forecasting purposes.

## Reproducibility package
- Date: 25-10-2024
- Author: Joris Depoortere (joris.depoortere@kuleuven.be)
- Structure: 
  - data: all the PV and weather data
  - notebooks: all the notebooks for the reproducibility check
  - src: all the source scripts

## Public Reproducibility package
- Date: 30-01-2025
- Updates:
  - The NL and BE PV data are proprietary and cannot be shared without 3rd party consent
  - The weather data has also been disabled because of GitHub data limits

## Requirements

Python 3.11.5

### Libraries
1. PyTorch 2.1.0
2. Scikit-learn 1.3.1
3. Numpy 1.24.1
4. Pandas 2.1.1 
5. matplotlib 3.8.0 
6. nbformat 5.9.2 
7. nbconvert 6.5.4
8. pickle 0.7.5 
9. xarray 2024.3.0 
10. ocf_blosc2 0.0.4
11. openmeteo_requests 1.2.0 
12. requests_cache 1.2.0 
13. retry_requests 2.0.0 
14. requests 2.28.1
15. json5 0.9.6
16. zarr 2.17.2

## Data

The data comes from several sources. But not all of them are available in the public package.

PV:
- AUS: https://github.com/pierre-haessig/ausgrid-solar-data?tab=readme-ov-file
- NL: Proprietary
- BE: Proprietary

Weather:
- https://huggingface.co/datasets/openclimatefix/dwd-icon-eu

## Reproducibility of figures and tables

The code repository has several notebooks which can be run to obtain the figures in our paper.
The below chapters explain how the notebooks work

### Base notebooks

The AUS base notebook executes the models and provides error metric tables for a specific system. 
The AUS notebook only result in the 'base' analysis without weather variables. 

The NL notebook has several configurations which can be set in the first cell of the notebook:
- With or without weather data included
- With a distance mismatch between source and target
- With a different seasonal periodicity
- Update 30/01/2025: This data is currently not publicly available and the notebooks has been disabled

The NL notebook specifies how to run these figures in the first cell of the notebook:
- Figure 3: All false
- Figure 4: weather_variables = True
- Table 1: seasonal = True
- Figure 5.a: distance = True
- Figure 5.b: distance = True, weather_variables = True
- Update 30/01/2025: This data is currently not publicly available and the notebooks has been disabled

For the AUS notebook we advise to use the run_multiple_instances notebook as discussed
in the next section.

### Run multiple instances

An additional notebook has been included to make it easy to loop over the base notebooks.
For the AUS notebook the user just has to specify 'AUS' and leave the analysis on 'base'. 

For the NL notebooks the user first has to specify the parameters in the base notebook
as explained in the previous section, and after that, specify 'NL' in the run_multiple_instances
notebook, as well as the type of analysis: 'base', 'distance' or 'seasonal'. After setting
these parameters, this notebook will run the analysis for all the systems in the dataset.
- Update 30/01/2025: This data is currently not publicly available and the notebooks has been disabled

Run multiple instances will sometimes output that an error occurred in the handling of one of the
base notebooks. This should only occur for the seasonality analysis, as some of the NL systems do 
not have data going back to a point specified by the seasonality analysis. If an error occurs in
other situations, something has gone wrong in the execution of the base notebook.
- Update 30/01/2025: This data is currently not publicly available and the notebooks has been disabled

The output for NL will produce numbered data: for example amstelveen_O. the output can jump to a
non-sequential number (for example from 4 to 6) this is because the initial dataset has data from
50 systems, but only 40 can be used for this research, so 10 systems in total are skipped.
- Update 30/01/2025: This data is currently not publicly available and the notebooks has been disabled

### figures

Once all the results have been obtained, the user can just execute the figures notebook and
obtain all the figures. The first cell has to be set for 
1. the location (NL, AUS or BE), and
2. the error metric (RMSE, MAE or MBE)

For AUS and BE only the figure 3 will be executed correctly, for NL all the code will be 
executed.

## Computer specs

The computer used to run the code is a HP ZBook Studio G9. It has the following key specs:
- CPU: Intel I9-12900
- RAM: 64 GB
- GPU: Nvidia RTX 3080 Ti Laptop

The runtime can be up to several days. A single location takes c. 20 minutes and a total of 340
locations were run over several different analysis. I would advise the user to take a sample of
the AUS dataset instead of the full 300 locations, which will save a lot of time. 

A (good, CUDA compatible) GPU is strongly advised to speed up the process.