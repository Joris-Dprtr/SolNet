# SolNet
Open-source tool to get a pre-trained deep learning model for solar forecasting purposes.

## Requirements

### Libraries
1. PyTorch
2. Sklearn
3. Numpy
4. Pandas

## Done

Base model:
1. User provides latitude and longitude
2. Source model gets returned

Updates:
1. Allow for GPU use
2. Additional locations close to source location can be added to further increase the amount of data included
3. Evaluation has been added to SolNet. The evaluation metrics + plots follow the following paper:

    Dazhi Yang, Stefano Alessandrini, Javier Antonanzas, Fernando Antonanzas-Torres, Viorel Badescu, Hans Georg Beyer, Robert Blaga, John Boland, Jamie M. Bright, Carlos F.M. Coimbra, Mathieu David, Âzeddine Frimane, Christian A. Gueymard, Tao Hong, Merlinde J. Kay, Sven Killinger, Jan Kleissl, Philippe Lauret, Elke Lorenz, Dennis van der Meer, Marius Paulescu, Richard Perez, Oscar Perpiñán-Lamigueiro, Ian Marius Peters, Gordon Reikard, David Renné, Yves-Marie Saint-Drenan, Yong Shuai, Ruben Urraca, Hadrien Verbois, Frank Vignola, Cyril Voyant, Jie Zhang,
    Verification of deterministic solar forecasts,
    Solar Energy,
    Volume 210,
    2020,
    Pages 20-37,
    ISSN 0038-092X,
    https://doi.org/10.1016/j.solener.2020.04.019.
    (https://www.sciencedirect.com/science/article/pii/S0038092X20303947)

4. Move from Darts to PyTorch. The complete code is now run in PyTorch directly
5. Use own API instead of PVLIB for fetching
6. A notebook reflecting the research done in the SolNet paper is added using AusGrid data: https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data
7. Access to weather data to include as features in the models

## Pipeline

### Short term
1. Additional models

### Long term
1. GUI
