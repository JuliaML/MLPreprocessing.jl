# MLPreprocessing

| **Package Status** | **Package Evaluator** | **Build Status**  |
|:------------------:|:---------------------:|:-----------------:|
| [![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md) | [![MLPreprocessing](http://pkg.julialang.org/badges/MLPreprocessing_0.5.svg)](http://pkg.julialang.org/?pkg=MLPreprocessing) [![MLPreprocessing](http://pkg.julialang.org/badges/MLPreprocessing_0.6.svg)](http://pkg.julialang.org/?pkg=MLPreprocessing) | [![Build Status](https://travis-ci.org/JuliaML/MLPreprocessing.jl.svg?branch=master)](https://travis-ci.org/JuliaML/MLPreprocessing.jl) [![AppVeyor](https://ci.appveyor.com/api/projects/status/80ns4uv5473kgj6f?svg=true)](https://ci.appveyor.com/project/Evizero/mlpreprocessing-jl) [![Coverage Status](https://coveralls.io/repos/github/JuliaML/MLPreprocessing.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaML/MLPreprocessing.jl?branch=master) |

## Overview

Utility package that provides end user friendly methods for feature scalings and polynomial
basis expansion. Feature scalings work on `Matrix`, `Vector` and `DataFrames`. It is possible to
have observations stored as columns or rows of a matrix. In order to distinguish between these cases
one can provide the parameter `obsdim`, where `obsdim=1` corresponds to "observations as rows" and 
`obsdim=2` to "observations as columns". Transformations can be computed on a subset
of columns/rows by defining a vector `operate_on`.

### StandardScaler
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

```julia
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
```

Note that for `transform!` the data matrix `X` has to be of type <: AbstractFloat
as the scaling occurs inplace. (E.g. cannot be of type Matrix{Int64}). This is not
the case for `transform` however.
For `DataFrames` `transform!` can be used on columns of type <: Integer.


### FixedRangeScaler
`FixedRangeScaler` is used with the functions `fit()`, `transform()` and `fit_transform()`
to scale data in a Matrix `X` or DataFrame to a fixed range [lower:upper].
After fitting a `FixedRangeScaler` to one data set, it can be used to perform the same
transformation to a new set of data. E.g. fit the `FixedRangeScaler` to your training
data and then apply the scaling to the test data at a later stage. (See examples below).

```julia
    fit(FixedRangeScaler, X[, lower, upper; obsdim, operate_on])

    fit_transform(FixedRangeScaler, X[, lower, upper; obsdim, operate_on])
```

`X`         :  Data of type Matrix or `DataFrame`.

`lower`     :  (Scalar) Lower limit of new range.
               Defaults to 0.

`upper`     :  (Scalar) Upper limit of new range.
               Defaults to 1.

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

```julia
    Xtrain = rand(100, 4)
    Xtest  = rand(10, 4)
    x = rand(10)
    D = DataFrame(A=rand(10), B=collect(1:10), C=[string(x) for x in 1:10])

    scaler = fit(FixedRangeScaler, Xtrain)
    scaler = fit(FixedRangeScaler, Xtrain, -1, 1)
    scaler = fit(FixedRangeScaler, Xtrain, -1, 1, obsdim=1)
    scaler = fit(FixedRangeScaler, Xtrain, -1, 1, obsdim=1, operate_on=[1,3])
    scaler = fit(FixedRangeScaler, D, -1, 1, operate_on=[:A,:B])

    Xscaled = transform(Xtest, scaler)
    transform!(Xtest, scaler)

    Xscaled, scaler = fit_transform(FixedRangeScaler, X, -1, 1, obsdim=1, operate_on=[1,2,4])
    scaler = fit_transform!(FixedRangeScaler, X, -1, 1, obsdim=1, operate_on=[1,2,4])
```

### Lower Level Functions
The lower level functions on which `StandardScaler` and `FixedRangeScaler` are built on can also
be used seperately.

#### center!()
```julia
    μ = center!(X[, μ; obsdim, operate_on])
```
Shift `X` along `obsdim` by `μ` according to X = X - μ
where `X` is of type Matrix or Vector and `D` of type DataFrame.

#### fixedrange!()
```julia
    lower, upper, xmin, xmax = fixedrange!(X[, lower, upper, xmin, xmax; obsdim, operate_on])
```
Normalize `X` or `D` along `obsdim` to the interval [lower:upper]
where `X` is of type Matrix or Vector and `D` of type DataFrame.
If `lower` and `upper`  are omitted the default range is [0:1].

#### standardize!()
```julia
    μ, σ = standardize!(X[, μ, σ; obsdim, operate_on])
```
Standardize `X` along `obsdim` according to X = (X - μ) / σ.
If μ and σ are omitted they are computed such that variables have a mean of zero.

### Polynomial Basis Expansion
```julia
    M = expand_poly(x[, degree=5, obsdim]) 
```
Perform a polynomial basis expansion of the given `degree` for the vector `x`.

```julia
julia> expand_poly(1:5, degree=3)
3×5 Array{Float64,2}:
 1.0  2.0   3.0   4.0    5.0
 1.0  4.0   9.0  16.0   25.0
 1.0  8.0  27.0  64.0  125.0

julia> expand_poly(1:5, degree=3, obsdim=1)
5×3 Array{Float64,2}:
 1.0   1.0    1.0
 2.0   4.0    8.0
 3.0   9.0   27.0
 4.0  16.0   64.0
 5.0  25.0  125.0

julia> expand_poly(1:5, 3, ObsDim.First()); # same but type-stable
```
