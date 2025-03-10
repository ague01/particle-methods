# Particle Methods projects
Collection of Particle Methods (PM) implementations for the PM course @ USI.

## Setup and run

**Prerequisites:** For all the projects the only requirements is to have a Python3 version installed and the [Pipenv](https://pipenv.pypa.io/en/latest/) module installed as

```
pip install --user pipenv
```

**Environment creation:** After cloning the repository, create the environment for the PM projects as

```
cd particle-methods
pipenv install
```

**Run:** After the environment is setted up you can execute the script by
```
pipenv run python script_name.py
```
In most cases when using IDES the environment is self detected so a plain run command `python script_name.py` is fine too.

In projects where there is more than one `.py` file, look at the project specific `README.md` file for the proper execution command.

## Projects

Each project is self contained in its folder, with its readme explaining results and eventually the execution steps when they differs from the ones proposed here.

Available Projects:

- `ising-model`: Monte Carlo simulation for the two-dimensional Ising Model.

