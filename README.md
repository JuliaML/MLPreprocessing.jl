# MLPreprocessing

| **Package Status** | **Package Evaluator** | **Build Status**  |
|:------------------:|:---------------------:|:-----------------:|
| [![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md) | [![MLPreprocessing](http://pkg.julialang.org/badges/MLPreprocessing_0.5.svg)](http://pkg.julialang.org/?pkg=MLPreprocessing) [![MLPreprocessing](http://pkg.julialang.org/badges/MLPreprocessing_0.6.svg)](http://pkg.julialang.org/?pkg=MLPreprocessing) | [![Build Status](https://travis-ci.org/JuliaML/MLPreprocessing.jl.svg?branch=master)](https://travis-ci.org/JuliaML/MLPreprocessing.jl) [![AppVeyor](https://ci.appveyor.com/api/projects/status/80ns4uv5473kgj6f?svg=true)](https://ci.appveyor.com/project/Evizero/mlpreprocessing-jl) [![Coverage Status](https://coveralls.io/repos/github/JuliaML/MLPreprocessing.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaML/MLPreprocessing.jl?branch=master) |

## Overview

Utility package that provides end user friendly methods for feature scalings and polynomial
basis expansion.

### Standardization
Standardization of data sets result in variables with a mean of 0 and variance of 1.
A common use case would be to fit a `StandardScaler` to the training data and later
apply the same transformation to the test data. `StandardScaler` is used with the 
functions `fit()`, `transform()` and `fit_transform()` as shown below.

```julia

    fit(StandardScaler, X[, μ, σ; obsdim, operate_on])

    fit_transform(StandardScaler, X[, μ, σ; obsdim, operate_on])
```

`X`         :  Data of type Matrix or `DataFrame`.

`μ`         :  Vector or scalar describing the translation.
               Defaults to mean(X, obsdim)

`σ`         :  Vector or scalar describing the scale.
               Defaults to std(X, obsdim)

`obsdim`    :  Specify which axis corresponds to observations.
               Defaults to obsdim=2 (observations are columns of matrix)
               For DataFrames `obsdim` is obsolete and rescaling occurs
               column wise.

`operate_on`:  Specify the indices of columns or rows to be centered.
               Defaults to all columns/rows.
               For DataFrames this must be a vector of symbols, not indices.
               E.g. `operate_on`=[1,3] will perform centering on columns
               with index 1 and 3 only (if obsdim=1, else rows 1 and 3)

Note on DataFrames:
Columns containing `NA` values are skipped.
Columns containing non numeric elements are skipped.

Examples:


    Xtrain = rand(100, 4)
    Xtest  = rand(10, 4)
    x = rand(4)
    Dtrain = DataFrame(A=rand(10), B=collect(1:10), C=[string(x) for x in 1:10])
    Dtest = DataFrame(A=rand(10), B=collect(1:10), C=[string(x) for x in 1:10])

    scaler = fit(StandardScaler, Xtrain)
    scaler = fit(StandardScaler, Xtrain, obsdim=1)
    scaler = fit(StandardScaler, Xtrain, obsdim=1, operate_on=[1,3])
    transform(Xtest, scaler)
    transform!(Xtest, scaler)
    transform(x, scaler)
    transform!(x, scaler)

    scaler = fit(StandardScaler, Dtrain)
    scaler = fit(StandardScaler, Dtrain, operate_on=[:A,:B])
    transform(Dtest, scaler)
    transform!(Dtest, scaler)

    Xscaled, scaler = fit_transform(StandardScaler, X, obsdim=1, operate_on=[1,2,4])
    scaler = fit_transform!(StandardScaler, X, obsdim=1, operate_on=[1,2,4])

Note that for `transform!` the data matrix `X` has to be of type <: AbstractFloat
as the scaling occurs inplace. (E.g. cannot be of type Matrix{Int64}). This is not
the case for `transform` however.
For `DataFrames` `transform!` can be used on columns of type <: Integer.
