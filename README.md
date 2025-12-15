# FSPB

## Installation

First you need to install the R environment (this you only need to do once):

```console
pixi run R
```

```R
install.packages("conformalInference.fd", repos="https://cloud.r-project.org")
```

## Reproducing the results

Run

```console
pixi run pytask
```

to reproduce the results, which will be written to the `bld` folder.


## R Implementation

A pure R implementation of the FSPB method, which also guided the development of this
Python implementation, is available at

https://github.com/creutzml/FunctionalPrediction

written by Michael L. Creutzinger.
