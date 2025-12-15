# FSPB

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

To reproduce the results, run

```console
pixi run pytask
```

which will create a `bld` folder containing the results.

---

> [!NOTE]
>  An **R implementation** of the FSPB method, which also guided the development of this
Python implementation, is available at
>
> https://github.com/creutzml/FunctionalPrediction
>
> written by Michael L. Creutzinger.
