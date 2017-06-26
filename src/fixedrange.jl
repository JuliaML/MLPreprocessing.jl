"""
    lower, upper, xmin, xmax = fixedrange!(X[, lower, upper, xmin, xmax; obsdim, operate_on])

or

    lower, upper, xmin, xmax = fixedrange!(D[, lower, upper, xmin, xmax; operate_on])


where `X` is of type Matrix or Vector and `D` of type DataFrame.
Normalize `X` or `D` along `obsdim` to the interval [lower:upper].
If `lower` and `upper`  are omitted the default range is [0:1].


`lower`     :  (Scalar) Lower limit of new range.
               Defaults to 0.

`upper`     :  (Scalar) Upper limit of new range.
               Defaults to 1.

`xmin`      :  (Vector) Minimum values of data before normalization. `xmin` will
               correspond to `lower` after transformation.
               Defaults to `minimum(X, obsdim)`.

`xmin`      :  (Vector) Maximum value of data before normalization. `xmax` will
               correspond to `upper` after transformation.
               Defaults to `maximum(X, obsdim)`.

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

    X = rand(4, 100)
    x = rand(10)
    D = DataFrame(A=rand(10), B=collect(1:10), C=[string(x) for x in 1:10])

    lower, upper, xmin, xmax = fixedrange!(X)
    lower, upper, xmin, xmax = fixedrange!(X, -1, 1)
    lower, upper, xmin, xmax = fixedrange!(X, -1, 1, obsdim=1)
    lower, upper, xmin, xmax = fixedrange!(X, -1, 1, obsdim=1, operate_on=[1,4])
    

    lower, upper, xmin, xmax = fixedrange!(D)
    lower, upper, xmin, xmax = fixedrange!(D, -1, 1)
    lower, upper, xmin, xmax = fixedrange!(D, -1, 1, operate_on=[:A,:B])
"""
function fixedrange!(X; obsdim=LearnBase.default_obsdim(X), operate_on=default_scaleselection(X, convert(ObsDimension, obsdim)))
    fixedrange!(X, convert(ObsDimension, obsdim), operate_on)
end

function fixedrange!{T,N}(X::AbstractArray{T,N}, ::ObsDim.Last, operate_on)
    fixedrange!(X, ObsDim.Constant{N}(), operate_on)
end

function fixedrange!{M}(X, obsdim::ObsDim.Constant{M}, operate_on)
    lower = 0
    upper = 1
    xmin = minimum(X, M)[operate_on]
    xmax = maximum(X, M)[operate_on]
    fixedrange!(X, lower, upper, xmin, xmax, obsdim, operate_on)
end

function fixedrange!(X, lower, upper; obsdim=LearnBase.default_obsdim(X), operate_on=default_scaleselection(X, convert(ObsDimension, obsdim)))
    fixedrange!(X, lower, upper, convert(ObsDimension, obsdim), operate_on)
end

function fixedrange!{M}(X, lower, upper, obsdim::ObsDim.Constant{M}, operate_on)
    xmin = minimum(X, M)[operate_on]
    xmax = maximum(X, M)[operate_on]
    fixedrange!(X, lower, upper, xmin, xmax, obsdim, operate_on)
end

function fixedrange!{T,M}(X::AbstractArray{T,M}, lower, upper, obsdim::ObsDim.Last, operate_on)
    fixedrange!(X, lower, upper, ObsDim.Constant{M}(), operate_on)
end

function fixedrange!(X, lower, upper, xmin, xmax; obsdim=LearnBase.default_obsdim(X), operate_on=default_scaleselection(X, convert(ObsDimension, obsdim)))
    fixedrange!(X, lower, upper, xmin, xmax, convert(ObsDimension, obsdim), operate_on)
end

function fixedrange!(X::AbstractMatrix, lower::Real, upper::Real, xmin::AbstractVector, xmax::AbstractVector, ::ObsDim.Constant{1}, operate_on::AbstractVector)
    @assert length(xmin) == length(xmax) == length(operate_on)
    xrange = xmax .- xmin
    scale = upper - lower
    nObs, nVars = size(X)

    for (i, iVar) in enumerate(operate_on)
        @inbounds for iObs in 1:nObs
            X[iObs, iVar] = lower + (X[iObs, iVar] - xmin[i]) / xrange[i] * scale
        end
    end
    lower, upper, xmin, xmax
end

function fixedrange!(X::AbstractMatrix, lower::Real, upper::Real, xmin::AbstractVector, xmax::AbstractVector, ::ObsDim.Constant{2}, operate_on::AbstractVector)
    @assert length(xmin) == length(xmax) == length(operate_on)
    xrange = xmax .- xmin
    scale = upper - lower
    nVars, nObs = size(X)

    for iObs in 1:nObs
        @inbounds for (i, iVar) in enumerate(operate_on)
            X[iVar, iObs] = lower + (X[iVar, iObs] - xmin[i]) / xrange[i] * scale
        end
    end
    lower, upper, xmin, xmax
end

function fixedrange!(X::AbstractMatrix, lower::Real, upper::Real, xmin::AbstractVector, xmax::AbstractVector, ::ObsDim.Last, operate_on::AbstractVector)
    fixedrange!(X, lower, upper, xmin, xmax, ObsDim.Constant{2}(), operate_on)
end

function fixedrange!(x::AbstractVector; lower=0.0, upper=1.0)
    xmin = minimum(x)
    xmax = maximum(x)
    fixedrange!(x, lower, upper, xmin, xmax)
end

function fixedrange!(x::AbstractVector, lower::Real, upper::Real)
    xmin = minimum(x)
    xmax = maximum(x)
    fixedrange!(x, lower, upper, xmin, xmax)
end

function fixedrange!{M}(x::AbstractVector, lower::Real, upper::Real, xmin::AbstractVector, xmax::AbstractVector, ::ObsDim.Constant{M}, operate_on::AbstractVector)
    @assert length(xmin) == length(xmax) == length(operate_on)
    xrange = xmax .- xmin
    scale = upper - lower
    nVars = length(x)
    @inbounds for (i, iVar) in enumerate(operate_on)
        x[iVar] = lower + (x[iVar] - xmin[i]) / xrange[i] * scale
    end
    lower, upper, xmin, xmax
end

function fixedrange!(x::AbstractVector, lower::Real, upper::Real, xmin::AbstractVector, xmax::AbstractVector, ::ObsDim.Last, operate_on::AbstractVector)
    fixedrange!(x, lower, upper, xmin, xmax, ObsDim.Constant{1}(), operate_on)
end

function fixedrange!(x::AbstractVector, lower::Real, upper::Real, xmin::Real, xmax::Real)
    xrange = xmax - xmin
    scale = upper - lower
    n = length(x)
    @inbounds for i in 1:n
        x[i] = lower + (x[i] - xmin) / xrange * scale
    end
    lower, upper, xmin, xmax
end

# --------------------------------------------------------------------

function fixedrange!(D::AbstractDataFrame; operate_on=default_scaleselection(D))
    fixedrange!(D, 0, 1, operate_on)
end

function fixedrange!(D::AbstractDataFrame, lower, upper; operate_on=default_scaleselection(D))
    fixedrange!(D, lower, upper, operate_on)
end

function fixedrange!(D::AbstractDataFrame, lower::Real, upper::Real, operate_on::AbstractArray)
    xmin = Float64[]
    xmax = Float64[]

    for colname in operate_on 
        if eltype(D[colname]) <: Real
            minval = minimum(D[colname])
            maxval = maximum(D[colname])
            if isna(minval)
                warn("Skipping \"$colname\" because it contains NA values")
                continue
            end
            fixedrange!(D, lower, upper, minval, maxval, colname)
            push!(xmin, minval)
            push!(xmax, maxval)
        else
            warn("Skipping \"$colname\" because data is not of type T <: Real.")
        end
    end
    lower, upper, xmin, xmax 
end

function fixedrange!(D::AbstractDataFrame, lower, upper, xmin, xmax; operate_on=default_scaleselection(D))
    fixedrange!(D, lower, upper, xmin, xmax, operate_on)
end

function fixedrange!(D::AbstractDataFrame, lower::Real, upper::Real, xmin::AbstractArray, xmax::AbstractArray, operate_on::AbstractVector)
    @assert length(xmin) == length(xmax) == length(operate_on)
    for (iVar, colname) in enumerate(operate_on)
        fixedrange!(D, lower, upper, xmin[iVar], xmax[iVar], colname)
    end
    lower, upper, xmin, xmax, operate_on 
end

function fixedrange!(D::AbstractDataFrame, lower::Real, upper::Real, xmin::Real, xmax::Real, colname::Symbol)
    if any(isna, D[colname]) | !(eltype(D[colname]) <: Real)
        warn("Skipping \"$colname\" because it contains NA values or is not of type <: Real")
    else
        newcol::Vector{Float64} = convert(Vector{Float64}, D[colname])
        fixedrange!(newcol, lower, upper, xmin, xmax)
        D[colname] = newcol
    end
    lower, upper, xmin, xmax, colname
end


"""
`FixedRangeScaler` is used with the functions `fit()`, `transform()` and `fit_transform()`
to scale data in a Matrix `X` or DataFrame to a fixed range [lower:upper].
After fitting a `FixedRangeScaler` to one data set, it can be used to perform the same
transformation to a new set of data. E.g. fit the `FixedRangeScaler` to your training
data and then apply the scaling to the test data at a later stage. (See examples below).

    fit(FixedRangeScaler, X[, lower, upper; obsdim, operate_on])

    fit_transform(FixedRangeScaler, X[, lower, upper; obsdim, operate_on])

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



Note that for `transform!` the data matrix `X` has to be of type <: AbstractFloat
as the scaling occurs inplace. (E.g. cannot be of type Matrix{Int64}). This is not
the case for `transform` however.
For `DataFrames` `transform!` can be used on columns of type <: Integer.
"""
immutable FixedRangeScaler{T<:Real,U<:Real,V<:Real,W<:Real,M,I}
    lower::T
    upper::U
    xmin::Vector{V}
    xmax::Vector{W}
    obsdim::ObsDim.Constant{M}
    operate_on::Vector{I}
end

function FixedRangeScaler{T<:Real,N}(X::AbstractArray{T,N}; obsdim=LearnBase.default_obsdim(X), operate_on=default_scaleselection(X, convert(ObsDimension, obsdim)))
    FixedRangeScaler(X, convert(ObsDimension, obsdim), operate_on)
end

function FixedRangeScaler{T<:Real,N,M}(X::AbstractArray{T,N}, obsdim::ObsDim.Constant{M}, operate_on)
    xmin = vec(minimum(X, M))[operate_on]
    xmax = vec(maximum(X, M))[operate_on]
    FixedRangeScaler(0, 1, xmin, xmax, obsdim, operate_on)
end

function FixedRangeScaler{T<:Real,N}(X::AbstractArray{T,N}, ::ObsDim.Last, operate_on)
    FixedRangeScaler(X, ObsDim.Constant{N}(), operate_on)
end

function FixedRangeScaler{T<:Real,N}(X::AbstractArray{T,N}, lower, upper; obsdim=LearnBase.default_obsdim(X), operate_on=default_scaleselection(X, convert(ObsDimension, obsdim)))
    FixedRangeScaler(X, lower, upper, convert(ObsDimension, obsdim), operate_on)
end

function FixedRangeScaler{T<:Real,N,M}(X::AbstractArray{T,N}, lower, upper, obsdim::ObsDim.Constant{M}, operate_on)
    xmin = vec(minimum(X, M))[operate_on]
    xmax = vec(maximum(X, M))[operate_on]
    FixedRangeScaler(lower, upper, xmin, xmax, obsdim, operate_on)
end

function FixedRangeScaler{T<:Real,N}(X::AbstractArray{T,N}, lower, upper, ::ObsDim.Last, operate_on)
    FixedRangeScaler(X, lower, upper, ObsDim.Constant{N}(), operate_on)
end

function FixedRangeScaler(D::AbstractDataFrame; operate_on=default_scaleselection(D))
    FixedRangeScaler(D, 0, 1, operate_on)
end

function FixedRangeScaler(D::AbstractDataFrame, lower::Real, upper::Real; operate_on=default_scaleselection(D))
    FixedRangeScaler(D, lower, upper, operate_on)
end

function FixedRangeScaler(D::AbstractDataFrame, lower::Real, upper::Real, operate_on::AbstractVector{Symbol})
    xmin = Float64[]
    xmax = Float64[]
    colnames = valid_columns(D, operate_on)
    for colname in colnames 
        push!(xmin, minimum(D[colname]))
        push!(xmax, maximum(D[colname]))
    end
    FixedRangeScaler(lower, upper, xmin, xmax, ObsDim.Constant{1}(), colnames)
end

function StatsBase.fit{T<:Real,N}(::Type{FixedRangeScaler}, X::AbstractArray{T,N}; obsdim=LearnBase.default_obsdim(X), operate_on=default_scaleselection(X, convert(ObsDimension, obsdim)))
    FixedRangeScaler(X, convert(ObsDimension, obsdim), operate_on)
end

function fit_transform{T<:Real,N}(::Type{FixedRangeScaler}, X::AbstractArray{T,N}; obsdim=LearnBase.default_obsdim(X), operate_on=default_scaleselection(X, convert(ObsDimension, obsdim)))
    scaler = FixedRangeScaler(X, convert(ObsDimension, obsdim), operate_on)
    Xnew = transform(X, scaler)
    return Xnew, scaler
end

function fit_transform!{T<:Real,N}(::Type{FixedRangeScaler}, X::AbstractArray{T,N}; obsdim=LearnBase.default_obsdim(X), operate_on=default_scaleselection(X, convert(ObsDimension, obsdim)))
    scaler = FixedRangeScaler(X, convert(ObsDimension, obsdim), operate_on)
    transform!(X, scaler)
    return scaler
end

function StatsBase.fit{T<:Real,N}(::Type{FixedRangeScaler}, X::AbstractArray{T,N}, lower, upper; obsdim=LearnBase.default_obsdim(X), operate_on=default_scaleselection(X, convert(ObsDimension, obsdim)))
    FixedRangeScaler(X, lower, upper, convert(ObsDimension, obsdim), operate_on)
end

function fit_transform{T<:Real,N}(::Type{FixedRangeScaler}, X::AbstractArray{T,N}, lower, upper; obsdim=LearnBase.default_obsdim(X), operate_on=default_scaleselection(X, convert(ObsDimension, obsdim)))
    scaler = FixedRangeScaler(X, lower, upper, convert(ObsDimension, obsdim), operate_on)
    Xnew = transform(X, scaler)
    return Xnew, scaler
end

function fit_transform!{T<:Real,N}(::Type{FixedRangeScaler}, X::AbstractArray{T,N}, lower, upper; obsdim=LearnBase.default_obsdim(X), operate_on=default_scaleselection(X, convert(ObsDimension, obsdim)))
    scaler = FixedRangeScaler(X, lower, upper, convert(ObsDimension, obsdim), operate_on)
    transform!(X, scaler)
    return scaler
end

function StatsBase.fit(::Type{FixedRangeScaler}, D::AbstractDataFrame; operate_on=default_scaleselection(D))
    FixedRangeScaler(D, 0, 1, operate_on)
end

function fit_transform(::Type{FixedRangeScaler}, D::AbstractDataFrame; operate_on=default_scaleselection(D))
    scaler = FixedRangeScaler(D, 0, 1, operate_on)
    Dnew = transform(D, scaler)
    return Dnew, scaler
end

function fit_transform!(::Type{FixedRangeScaler}, D::AbstractDataFrame; operate_on=default_scaleselection(D))
    scaler = FixedRangeScaler(D, 0, 1, operate_on)
    transform!(D, scaler)
    return scaler
end

function StatsBase.fit(::Type{FixedRangeScaler}, D::AbstractDataFrame, lower, upper; operate_on=default_scaleselection(D))
    FixedRangeScaler(D, lower, upper, operate_on)
end

function fit_transform(::Type{FixedRangeScaler}, D::AbstractDataFrame, lower, upper; operate_on=default_scaleselection(D))
    scaler = FixedRangeScaler(D, lower, upper, operate_on)
    Dnew = transform(D, scaler)
    return Dnew, scaler
end

function fit_transform!(::Type{FixedRangeScaler}, D::AbstractDataFrame, lower, upper; operate_on=default_scaleselection(D))
    scaler = FixedRangeScaler(D, lower, upper, operate_on)
    transform!(D, scaler)
    return scaler
end

function transform!{T<:AbstractFloat,N}(X::AbstractArray{T,N}, cs::FixedRangeScaler)
    fixedrange!(X, cs.lower, cs.upper, cs.xmin, cs.xmax, cs.obsdim, cs.operate_on)
end

function transform!{T<:AbstractFloat}(x::AbstractVector{T}, cs::FixedRangeScaler)
    fixedrange!(x, cs.lower, cs.upper, cs.xmin, cs.xmax, cs.obsdim, cs.operate_on)
end

function transform!(D::AbstractDataFrame, cs::FixedRangeScaler)
    fixedrange!(D, cs.lower, cs.upper, cs.xmin, cs.xmax, cs.operate_on)
end

function transform{T<:AbstractFloat,N}(X::AbstractArray{T,N}, cs::FixedRangeScaler)
    Xnew = copy(X)
    fixedrange!(Xnew, cs.lower, cs.upper, cs.xmin, cs.xmax, cs.obsdim, cs.operate_on)
    Xnew
end

function transform{T<:Real,N}(X::AbstractArray{T,N}, cs::FixedRangeScaler)
    Xnew = convert(AbstractArray{Float64, N}, X)
    fixedrange!(Xnew, cs.lower, cs.upper, cs.xmin, cs.xmax, cs.obsdim, cs.operate_on)
    Xnew
end

function transform(D::AbstractDataFrame, cs::FixedRangeScaler)
    Dnew = deepcopy(D)
    fixedrange!(Dnew, cs.lower, cs.upper, cs.xmin, cs.xmax, cs.operate_on)
    Dnew
end
