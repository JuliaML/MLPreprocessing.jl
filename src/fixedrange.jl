"""
    lower, upper, xmin, xmax = fixedrange!(X[, lower, upper, xmin, xmax; obsdim])

Rescale `X` to the interval (lower:upper) along `obsdim`. If `upper` and `lower` are not
provided they default to 0 and 1 respectively, rescaling the data to the unit range (0:1).
`xmin` and `xmax` are vectors consisiting of the maximum and minimum values of `X` along obsdim.
`xmin`, `xmax` default to minimum(X, obsdim) and maximum(X, obsdim) respectively.
`obsdim` refers to the dimension of observations, e.g. `obsdim`=1 if the rows of `X` correspond to
measurements. `obsdim`=2 if columns of `X` represent measurements.

Examples:

    X = rand(10, 4)

    fixedrange!(X, obsdim=1)
    fixedrange!(X, -1, 1, obsdim=2)


where `X` is of type Matrix or Vector and `D` of type DataFrame.

Center `X` along `obsdim` around the corresponding entry in the
vector `μ`.


`μ`         :  Vector or value describing the center.
               Defaults to mean(X, 2)

`obsdim`    :  Specify which axis corresponds to observations.
               Defaults to obsdim=2 (observations are columns of matrix)
               For DataFrames `obsdim` is obsolete and centering occurs
               column wise.

`operate_on`:  Specify the indices of columns or rows to be centered.
               Defaults to all columns/rows.
               For DataFrames this must be a vector of symbols, not indices
               E.g. `operate_on`=[1,3] will perform centering on columns
               with index 1 and 3 only (if obsdim=1, else rows 1 and 3)


Note on DataFrames:
Columns containing `NA` values are skipped.
Columns containing non numeric elements are skipped.

Examples:

    X = rand(4, 100)
    x = rand(10)
    D = DataFrame(A=rand(10), B=collect(1:10), C=[string(x) for x in 1:10])

    μ = center!(X, obsdim=2)
    μ = center!(X, ObsDim.First())
    μ = center!(X, obsdim=1, operate_on=[1,3]
    μ = center!(X, [7.0, 8.0], obsdim=1, operate_on=[1,3]
    μ = center!(D)
    μ = center!(D, operate_on=[:A, :B])
    μ = center!(D, [-1,-1], operate_on=[:A, :B])
"""

function fixedrange!(X; obsdim=LearnBase.default_obsdim(X), operate_on=default_scalerange(X, convert(ObsDimension, obsdim)))
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

function fixedrange!(X, lower, upper; obsdim=LearnBase.default_obsdim(X), operate_on=default_scalerange(X, convert(ObsDimension, obsdim)))
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

function fixedrange!(X, lower, upper, xmin, xmax; obsdim=LearnBase.default_obsdim(X), operate_on=default_scalerange(X, convert(ObsDimension, obsdim)))
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

function fixedrange!{T,M}(X::AbstractArray{T,M}, lower::Real, upper::Real, xmin::Real, xmax::Real, ::ObsDim.Last, operate_on::AbstractVector)
    fixedrange!(X, lower, upper, xmin, xmax, ObsDim.Constant{M}(), operate_on)
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

function fixedrange!(D::AbstractDataFrame; operate_on=default_scalerange(D))
    fixedrange!(D, 0, 1, operate_on)
end

function fixedrange!(D::AbstractDataFrame, lower, upper; operate_on=default_scalerange(D))
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

function fixedrange!(D::AbstractDataFrame, lower, upper, xmin, xmax; operate_on=default_scalerange(D))
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
    if any(isna(D[colname])) | !(eltype(D[colname]) <: Real)
        warn("Skipping \"$colname\" because it contains NA values or is not of type <: Real")
    else
        newcol::Vector{Float64} = convert(Vector{Float64}, D[colname])
        fixedrange!(newcol, lower, upper, xmin, xmax)
        D[colname] = newcol
    end
    lower, upper, xmin, xmax, colname
end

immutable FixedRangeScaler{T<:Real,U<:Real,V<:Real,W<:Real,M,I}
    lower::T
    upper::U
    xmin::Vector{V}
    xmax::Vector{W}
    obsdim::ObsDim.Constant{M}
    operate_on::Vector{I}
end

function FixedRangeScaler{T<:Real,N}(X::AbstractArray{T,N}; obsdim=LearnBase.default_obsdim(X), operate_on=default_scalerange(X, convert(ObsDimension, obsdim)))
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

function FixedRangeScaler{T<:Real,N}(X::AbstractArray{T,N}, lower, upper; obsdim=LearnBase.default_obsdim(X), operate_on=default_scalerange(X, convert(ObsDimension, obsdim)))
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

function FixedRangeScaler(D::AbstractDataFrame; operate_on=default_scalerange(D))
    FixedRangeScaler(D, 0, 1, operate_on)
end

function FixedRangeScaler(D::AbstractDataFrame, lower::Real, upper::Real;  operate_on=default_scalerange(D))
    FixedRangeScaler(D, lower, upper, operate_on)
end

function FixedRangeScaler(D::AbstractDataFrame, lower::Real, upper::Real,  operate_on::AbstractVector{Symbol})
    xmin = Float64[]
    xmax = Float64[]
    for colname in operate_on 
        push!(xmin, minimum(D[colname]))
        push!(xmax, maximum(D[colname]))
    end
    FixedRangeScaler(lower, upper, xmin, xmax, ObsDim.Constant{1}(), operate_on)
end

function valid_columns(D::AbstractDataFrame)
    valid_colnames = Symbol[]
    for colname in names(D)
        if (eltype(D[colname]) <: Real) & !(any(isnull(D[colname])))
            push!(valid_colnames, colname)
        else
            warn("Skipping \"$colname\" because it either contains NA or is not of type <: Real")
        end
    end
    valid_colnames
end

function StatsBase.fit{T<:Real,N}(::Type{FixedRangeScaler}, X::AbstractArray{T,N}; obsdim=LearnBase.default_obsdim(X), operate_on=default_scalerange(X, convert(ObsDimension, obsdim)))
    FixedRangeScaler(X, convert(ObsDimension, obsdim), operate_on)
end

function StatsBase.fit{T<:Real,N}(::Type{FixedRangeScaler}, X::AbstractArray{T,N}, lower, upper; obsdim=LearnBase.default_obsdim(X), operate_on=default_scalerange(X, convert(ObsDimension, obsdim)))
    FixedRangeScaler(X, lower, upper, convert(ObsDimension, obsdim), operate_on)
end

function StatsBase.fit(::Type{FixedRangeScaler}, D::AbstractDataFrame; operate_on=default_scalerange(D))
    FixedRangeScaler(D, 0, 1, operate_on)
end

function StatsBase.fit(::Type{FixedRangeScaler}, D::AbstractDataFrame, lower, upper; operate_on=default_scalerange(D))
    FixedRangeScaler(D, lower, upper, operate_on)
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
