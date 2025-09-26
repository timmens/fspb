# FSPB

## Installation

> [!IMPORTANT]
> Currently, the environment can only be installed on Linux and Intel-based
> MacOS.

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
