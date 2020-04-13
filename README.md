# Bayesian_regression_pyro
This repo contains working examples MCMC and SVI models for bayesian regression using the pyro library. Goal is to have flexible easy to use classes with a syntax similar to scikit-learn fit /predict.


Algorithm | Description | Class 
--- | --- | --- 
HMC MCMC with NUTS | Hamilton Monte Carlo Markov Chain Monte Carlo Method with No U-Turn Sample | bay_reg_mcmc
SVI | Stochastic Variational Inference | bay_reg_svi

# To do
* [ ] Implement non linear use case
* [ ] Implement non linear use case with previous transformation
* [ ] Implement use case for multidimensional input


# File and folder description
File/Folder| Description 
--- | ---
example.ipynb | Example use of those two algorithms
bayesian_regression_mcmc_svi.py | bay_reg_mcmc & bay_reg_svi class

# Installation 
```bash
pip install -r requirements.txt
```

## If you can want to run the file in a new enviroment:
- Make sure conda is installed (Best practice, set up with virtualenv is not tested)
- Open a terminal or a anaconda prompt
- If desired make new enviroment: conda create -n name_of_enviroment python
- Activate enviroment conda activate: conda create name_of_enviroment
- Install dependencies: pip install requirements.txt
- If the new enviroment / kernel is supposed to be used in Jupyter, install kernel:
```bash
    python -m ipykernel install --name name_of_enviroment
```
- Open your Jupyter Notebook it should work now


