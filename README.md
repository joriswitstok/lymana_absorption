# Lyman-alpha absorption fitting
<br>
<img src="/IGM_DLA_absorption.png" width="100%">
<br>

## Contents
1. [Description](#Description)

## <a name="Description"></a>Description

This is a code that uses the pymultinest package ([Feroz et al. 2009](); [Buchner et al. 2014](https://ui.adsabs.harvard.edu/abs/2014A%26A...564A.125B/abstract)) to fit Lyman-alpha (Lyα) damping-wing absorption, both from the IGM and local damped-Lyα absorbing (DLA) systems. The main functionality has been described in [D'Eugenio et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023arXiv231109908D/abstract). Below, its usage is illustrated with examples.

## <a name="Installation"></a>Installation and setup

### <a name="Cloning"></a>Cloning

First, obtain and install the latest version of the code, which can be done via `pip`. Alternatively, you can clone the repository by navigating to your desired installation folder and using

```
git clone https://github.com/joriswitstok/lymana_absorption.git
```

### <a name="Package_requirements"></a>Package requirements

Running the code requires the following Python packages:
- `numpy`
- `scipy`
- `astropy`
- `emcee`
- `mpi4py`
- `pymultinest`
- `spectres`
- `corner`
- `matplotlib`
- `seaborn`
- `mock`
  
Most of these modules are easily installed via the file `lya3.yml` provided in the main folder, which can be used to create an `conda` environment in Python 3 (see the `conda` [documentation on environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more details) containing all the required packages. However, the `lymana_absorption` code and several MPI-related packages (`mpi4py` and `pymultinest`) should be installed via `pip` to ensure properly linking to a local MPI installation (if available).

If you have `conda` installed and set up as a `python` distribution, this can be achieved with:

```
conda env create -f lya3.yml
```

Before running the code, the environment needs to be activated using

```
conda activate lya3
```

By default, the terminal will indicate the environment is active by showing a prompt similar to:

```
(lya3) $ 
```

The `lymana_absorption` code is then easily installed via `pip`:

```
(lya3) $ pip3 install lymana_absorption
```

## <a name="Example_usage"></a>Example usage

### <a name="Running_the_plotting_script"></a>Running a test script to obtain transmission curves

This section goes through an example usage case of `lymana_absorption` by running the file `example_plots.py` (located in the `examples` folder). The first step is to activate the environment as explained in [the previous section](#Package_requirements). So, starting from the main folder, the script would be run as follows:

```
$ conda activate lya3
(lya3) $ cd lymana_absorption/examples/
(lya3) $ python example_plots.py
```

If it has finished successfully, several figures illustrating IGM and DLA transmission curves will have been saved in the `examples` folder.

### <a name="Running_the_fitting_script"></a>Running a test script to fit observed damping-wing absorption

A more advanced example usage case is illustrated by running the file `example_fit.py` (again located in the `examples` folder). This script performs a fitting routine to the observed spectrum of GS-z13 (as in [Hainline et al. 2024](https://ui.adsabs.harvard.edu/abs/2023arXiv231109908D/abstract)), given a resolution curve and an intrinsic model spectrum.