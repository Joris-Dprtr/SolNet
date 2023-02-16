# SolNet
Open-source tool to get a pre-trained deep learning model for solar forecasting purposes.

## Requirements

### Libraries
TBD

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

## Pipeline

### Short term
1. Allow user to specify features
2. Finetuning functionality

### Long term
1. Additional models
2. Move away from some 3rd party libraries (f.e. PVLIB, Darts? (move to PyTorch directly))
3. GUI