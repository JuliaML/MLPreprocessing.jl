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

"""

function fixedrange!(X; obsdim=LearnBase.default_obsdim(X))
    fixedrange!(X, convert(ObsDimension, obsdim))
end

function fixedrange!{T,N}(X::AbstractArray{T,N}, ::ObsDim.Last)
    fixedrange!(X, ObsDim.Constant{N}())
end

function fixedrange!{M}(X, obsdim::ObsDim.Constant{M})
    lower = 0
    upper = 1
    xmin = minimum(X, M)
    xmax = maximum(X, M)
    fixedrange!(X, lower, upper, xmin, xmax, obsdim)
end

function fixedrange!(X, lower, upper; obsdim=LearnBase.default_obsdim(X))
    fixedrange!(X, lower, upper, convert(ObsDimension, obsdim))
end

function fixedrange!{M}(X, lower, upper, obsdim::ObsDim.Constant{M})
    xmin = minimum(X, M)
    xmax = maximum(X, M)
    fixedrange!(X, lower, upper, xmin, xmax, obsdim)
end

function fixedrange!(X, lower, upper, xmin, xmax; obsdim=LearnBase.default_obsdim(X))
    fixedrange!(X, lower, upper, xmin, xmax, convert(ObsDimension, obsdim))
end

function fixedrange!(X::AbstractMatrix, lower, upper, xmin, xmax, ::ObsDim.Constant{1})
    xrange = xmax .- xmin
    scale = upper - lower
    nObs, nVars = size(X)

    for iVar in 1:nVars
        @inbounds for iObs in 1:nObs
            X[iObs, iVar] = lower + (X[iObs, iVar] - xmin[iVar]) / xrange[iVar] * scale
        end
    end
    lower, upper, xmin, xmax
end

function fixedrange!(X::AbstractMatrix, lower, upper, xmin, xmax, ::ObsDim.Constant{2})
    xrange = xmax .- xmin
    scale = upper - lower
    nVars, nObs = size(X)

    for iObs in 1:nObs
        @inbounds for iVar in 1:nVars
            X[iVar, iObs] = lower + (X[iVar, iObs] - xmin[iVar]) / xrange[iVar] * scale
        end
    end
    lower, upper, xmin, xmax
end

function fixedrange!{N}(X::AbstractVector, lower::Real, upper::Real, xmin::Vector, xmax::Vector, ::ObsDim.Constant{N})
    @assert length(xmin) == length(xmax) == length(X) 
    xrange = xmax .- xmin
    scale = upper - lower
    nVars = length(X)

    @inbounds for iVar in eachindex(X) 
        X[iVar] = lower + (X[iVar] - xmin[iVar]) / xrange[iVar] * scale
    end
    lower, upper, xmin, xmax
end

function fixedrange!{N}(X::AbstractVector, lower::Real, upper::Real, xmin::Real, xmax::Real, ::ObsDim.Constant{N})
    xrange = xmax - xmin
    scale = upper - lower
    nVars = length(X)

    @inbounds for iVar in eachindex(X) 
        X[iVar] = lower + (X[iVar] - xmin) / xrange * scale
    end
    lower, upper, xmin, xmax
end


immutable FixedRangeScaler 
    lower::Float64
    upper::Float64
    xmin::Vector
    xmax::Vector
    obsdim::ObsDim.Constant{}

    function FixedRangeScaler(lower, upper, xmin, xmax, obsdim)
        @assert length(xmin) == length(xmax) 
        new(lower, upper, xmin, xmax, convert(ObsDimension, obsdim))
    end
end


function FixedRangeScaler{T<:Real}(X::AbstractArray{T}; obsdim=LearnBase.default_obsdim(X))
    FixedRangeScaler(X, convert(ObsDimension, obsdim))
end

function FixedRangeScaler{T<:Real,M}(X::AbstractArray{T,M}, ::ObsDim.Last)
    FixedRangeScaler(X, ObsDim.Constant{M}())
end

function FixedRangeScaler{T<:Real,M}(X::AbstractArray{T}, obsdim::ObsDim.Constant{M})
    FixedRangeScaler(0, 1, vec(minimum(X, M)), vec(maximum(X, M)), obsdim)
end

function FixedRangeScaler{T<:Real}(X::AbstractArray{T}, lower, upper; obsdim=LearnBase.default_obsdim(X))
    FixedRangeScaler(X, lower, upper, convert(ObsDimension, obsdim))
end

function FixedRangeScaler{T<:Real,M}(X::AbstractMatrix{T}, lower, upper, obsdim::ObsDim.Constant{M})
    FixedRangeScaler(lower, upper, vec(minimum(X, M)), vec(maximum(X, M)), obsdim)
end

function FixedRangeScaler{T<:Real,M}(X::AbstractArray{T,M}, lower, upper, ::ObsDim.Last)
    FixedRangeScaler(X, lower, upper, ObsDim.Constant{M}())
end

function StatsBase.fit{T<:Real}(::Type{FixedRangeScaler}, X::AbstractArray{T}; obsdim=LearnBase.default_obsdim(X))
    FixedRangeScaler(X, obsdim=obsdim)
end

function StatsBase.fit{T<:Real}(::Type{FixedRangeScaler}, X::AbstractArray{T}, lower, upper; obsdim=LearnBase.default_obsdim(X))
    FixedRangeScaler(X, lower, upper, obsdim=obsdim)
end

function transform!{T<:AbstractFloat}(X::AbstractArray{T}, cs::FixedRangeScaler)
    unitrange!(X, cs.lower, cs.upper, cs.xmin, cs.xmax, cs.obsdim)
end

function transform{T<:AbstractFloat}(X::AbstractArray{T}, cs::FixedRangeScaler)
    Xnew = copy(X)
    unitrange!(Xnew, cs.lower, cs.upper, cs.xmin, cs.xmax, cs.obsdim)
    Xnew
end
