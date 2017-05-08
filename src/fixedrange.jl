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

function fixedrange!(X::AbstractMatrix, lower::Real, upper::Real, xmin::AbstractVector, xmax::AbstractVector, ::ObsDim.Constant{1})
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

function fixedrange!(X::AbstractMatrix, lower::Real, upper::Real, xmin::AbstractVector, xmax::AbstractVector, ::ObsDim.Constant{2})
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

function fixedrange!{M}(X::AbstractVector, lower::Real, upper::Real, xmin::AbstractVector, xmax::AbstractVector, ::ObsDim.Constant{M})
    xrange = xmax .- xmin
    scale = upper - lower
    nVars = length(X)

    @inbounds for iVar in eachindex(X) 
        X[iVar] = lower + (X[iVar] - xmin[iVar]) / xrange[iVar] * scale
    end
    lower, upper, xmin, xmax
end

immutable FixedRangeScaler{T<:Real,U<:Real,V<:Real,W<:Real,M}
    lower::T
    upper::U
    xmin::Vector{V}
    xmax::Vector{W}
    obsdim::ObsDim.Constant{M}
end

function FixedRangeScaler{T<:Real,N}(X::AbstractArray{T,N}; obsdim=LearnBase.default_obsdim(X))
    FixedRangeScaler(X, convert(ObsDimension, obsdim))
end

function FixedRangeScaler{T<:Real,N,M}(X::AbstractArray{T,N}, obsdim::ObsDim.Constant{M})
    FixedRangeScaler(0, 1, vec(minimum(X, M)), vec(maximum(X, M)), obsdim)
end

function FixedRangeScaler{T<:Real,N}(X::AbstractArray{T,N}, ::ObsDim.Last)
    FixedRangeScaler(X, ObsDim.Constant{N}())
end

function FixedRangeScaler{T<:Real,N}(X::AbstractArray{T,N}, lower, upper; obsdim=LearnBase.default_obsdim(X))
    FixedRangeScaler(X, lower, upper, convert(ObsDimension, obsdim))
end

function FixedRangeScaler{T<:Real,N,M}(X::AbstractArray{T,N}, lower, upper, obsdim::ObsDim.Constant{M})
    FixedRangeScaler(lower, upper, vec(minimum(X, M)), vec(maximum(X, M)), obsdim)
end

function FixedRangeScaler{T<:Real,N}(X::AbstractArray{T,N}, lower, upper, ::ObsDim.Last)
    FixedRangeScaler(X, lower, upper, ObsDim.Constant{N}())
end

function StatsBase.fit{T<:Real,N}(::Type{FixedRangeScaler}, X::AbstractArray{T,N}; obsdim=LearnBase.default_obsdim(X))
    FixedRangeScaler(X, convert(ObsDimension, obsdim))
end

function StatsBase.fit{T<:Real,N}(::Type{FixedRangeScaler}, X::AbstractArray{T,N}, lower, upper; obsdim=LearnBase.default_obsdim(X))
    FixedRangeScaler(X, lower, upper, convert(ObsDimension, obsdim))
end

function transform!{T<:AbstractFloat,N}(X::AbstractArray{T,N}, cs::FixedRangeScaler)
    fixedrange!(X, cs.lower, cs.upper, cs.xmin, cs.xmax, cs.obsdim)
end

function transform{T<:AbstractFloat,N}(X::AbstractArray{T,N}, cs::FixedRangeScaler)
    Xnew = copy(X)
    fixedrange!(Xnew, cs.lower, cs.upper, cs.xmin, cs.xmax, cs.obsdim)
    Xnew
end

function transform{T<:Real,N}(X::AbstractArray{T,N}, cs::FixedRangeScaler)
    Xnew = convert(AbstractArray{Float64, N}, X)
    fixedrange!(Xnew, cs.lower, cs.upper, cs.xmin, cs.xmax, cs.obsdim)
    Xnew
end
