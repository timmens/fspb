# FSPB

Repository containing the reproduction code for the paper *"Adaptive Simultaneous
Prediction Bands in Concurrent Functional Linear Regression"* by Michael L. Creutzinger,
Dominik Liebl, Tim Mensinger, and Julia L. Sharp.

## Installation

First you need to install the R environment (this you only need to do once):

```console
pixi run R
```

which will open the R console. Then run:

```R
install.packages("conformalInference.fd", repos="https://cloud.r-project.org")
```

## Reproducing the results

To reproduce the results, set the number of cores (`N_JOBS`) in `src/fspb/config.py`,
and run

```console
pixi run pytask
```

which will create a `paper_bld` folder containing the results from the paper. All
intermediate results can be found in the `bld` folder.


> [!WARNING]
> The results will be different from those in the paper, since the data that is used
> here is anonymized, while the paper used the original data.

---

> [!NOTE]
>  An **R implementation** of the FSPB method, which also guided the development of this
Python implementation, is available at
>
> https://github.com/creutzml/FunctionalPrediction
>
> written by Michael L. Creutzinger.
