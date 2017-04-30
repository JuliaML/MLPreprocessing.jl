"""
    μ, σ = standardize!(X[, μ, σ, obsdim])

Center `X` along `obsdim` around the corresponding entry in the
vector `μ` and then standardize each feature using the corresponding
entry in the vector `σ`.
"""
function standardize!(X, μ, σ; obsdim=LearnBase.default_obsdim(X))
    standardize!(X, μ, σ, convert(ObsDimension, obsdim))
end

function standardize!{T,N}(X::AbstractArray{T,N}, μ, σ, ::ObsDim.Last)
    standardize!(X, μ, σ, ObsDim.Constant{N}())
end

function standardize!(X; obsdim=LearnBase.default_obsdim(X))
    standardize!(X, convert(ObsDimension, obsdim))
end

function standardize!{T,N}(X::AbstractArray{T,N}, ::ObsDim.Last)
    standardize!(X, ObsDim.Constant{N}())
end

function standardize!{T,N,M}(X::AbstractArray{T,N}, obsdim::ObsDim.Constant{M})
    μ = vec(mean(X, M))
    σ = vec(std(X, M))
    standardize!(X, μ, σ, obsdim)
end

function standardize!(X::AbstractVector, ::ObsDim.Constant{1})
    μ = mean(X)
    σ = std(X)
    for i in 1:length(X)
        X[i] = (X[i] - μ) / σ
    end
    μ, σ
end

function standardize!(X::AbstractMatrix, μ::AbstractVector, σ::AbstractVector, ::ObsDim.Constant{2})
    σ[σ .== 0] = 1
    nVars, nObs = size(X)
    for iObs in 1:nObs
        @inbounds for iVar in 1:nVars
            X[iVar, iObs] = (X[iVar, iObs] - μ[iVar]) / σ[iVar]
        end
    end
    μ, σ
end

function standardize!(X::AbstractMatrix, μ::AbstractVector, σ::AbstractVector, ::ObsDim.Constant{1})
    σ[σ .== 0] = 1
    nObs, nVars = size(X)
    for iVar in 1:nVars
        @inbounds for iObs in 1:nObs
            X[iObs, iVar] = (X[iObs, iVar] - μ[iVar]) / σ[iVar]
        end
    end
    μ, σ
end

function standardize!(X::AbstractVector, μ::AbstractVector, σ::AbstractVector, ::ObsDim.Constant{1})
    @inbounds for i in 1:length(X)
        X[i] = (X[i] - μ[i]) / σ[i]
    end
    μ, σ
end

function standardize!(X::AbstractVector, μ::AbstractFloat, σ::AbstractFloat, ::ObsDim.Constant{1})
    @inbounds for i in 1:length(X)
        X[i] = (X[i] - μ) / σ
    end
    μ, σ
end

immutable StandardScaler 
    offset::Vector{Float64}
    scale::Vector{Float64}
    obsdim::ObsDim.Constant{}

    function StandardScaler(offset, scale, obsdim)
        @assert length(offset) == length(scale) 
        new(offset, scale, convert(ObsDimension, obsdim))
    end
end

function StandardScaler{T<:Real}(X::AbstractMatrix{T}; obsdim=LearnBase.default_obsdim(X))
    StandardScaler(X, convert(ObsDimension, obsdim))
end

function StandardScaler{T<:Real,M}(X::AbstractArray{T,M}, ::ObsDim.Last)
    StandardScaler(X, ObsDim.Constant{M})
end

function StandardScaler{T<:Real,M}(X::AbstractArray{T,M}, obsdim::ObsDim.Constant{M})
    StandardScaler(mean(X, M), std(X, M), obsdim)
end

function StandardScaler{T<:Real,M}(X::AbstractArray{T,M}, offset, scale; obsdim=LearnBase.default_obsdim(X))
    StandardScaler(offset, scale, convert(ObsDimension, obsdim))
end

function StandardScaler{T<:Real,M}(X::AbstractArray{T,M}, offset, scale, ::ObsDim.Last)
    StandardScaler(offset, scale, ObsDim.Constant{M})
end

function StatsBase.fit{T<:Real}(::Type{StandardScaler}, X::AbstractMatrix{T}; obsdim=LearnBase.default_obsdim(X))
    StandardScaler(X, obsdim=obsdim)
end

function transform!{T<:Real}(cs::StandardScaler, X::AbstractMatrix{T})
    @assert length(cs.offset) == size(X, 1)
    standardize!(X, cs.offset, cs.scale, obsdim=cs.obsdim)
    X
end

function transform{T<:AbstractFloat}(cs::StandardScaler, X::AbstractMatrix{T})
    Xnew = copy(X)
    transform!(cs, Xnew)
end

function transform{T<:Real}(cs::StandardScaler, X::AbstractMatrix{T})
    X = convert(AbstractMatrix{AbstractFloat}, X)
    transform!(cs, X)
end
