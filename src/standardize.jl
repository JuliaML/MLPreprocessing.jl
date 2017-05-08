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

# --------------------------------------------------------------------

function standardize!(D::AbstractDataFrame)
    standardize!(D, names(D))
end

function standardize!(D::AbstractDataFrame, colnames::AbstractVector{Symbol})
    μ_vec = Float64[]
    σ_vec = Float64[]

    for colname in colnames
        if eltype(D[colname]) <: Real
            μ = mean(D[colname])
            σ = std(D[colname])
            if isna(μ)
                warn("Column \"$colname\" contains NA values, skipping rescaling of this column!")
                continue
            end
            standardize!(D, colname, μ, σ)
            push!(μ_vec, μ)
            push!(σ_vec, σ)
        else
            warn("Skipping \"$colname\", rescaling only valid for columns of type T <: Real.")
        end
    end
    μ_vec, σ_vec
end

function standardize!(D::AbstractDataFrame, colnames::AbstractVector{Symbol}, μ::AbstractVector, σ::AbstractVector)
    for (icol, colname) in enumerate(colnames)
        if eltype(D[colname]) <: Real
            standardize!(D, colname, μ[icol], σ[icol])
        else
            warn("Skipping \"$colname\", rescaling only valid for columns of type T <: Real.")
        end
    end
    μ, σ
end

function standardize!(D::AbstractDataFrame, colname::Symbol, μ::Real, σ::Real)
    if sum(isna(D[colname])) > 0
        warn("Column \"$colname\" contains NA values, skipping rescaling on this column!")
    else
        newcol::Vector{Float64} = convert(Vector{Float64}, D[colname])
        nobs = length(newcol)
        @inbounds for i in eachindex(newcol)
            newcol[i] = (newcol[i] - μ) / σ
       end
        D[colname] = newcol
    end
    μ, σ
end


immutable StandardScaler{T,U,M}
    offset::Vector{T}
    scale::Vector{U}
    obsdim::ObsDim.Constant{M}
end

function StandardScaler{T<:Real,M}(X::AbstractArray{T,M}; obsdim=LearnBase.default_obsdim(X))
    StandardScaler(X, convert(ObsDimension, obsdim))
end

function StandardScaler{T<:Real,N,M}(X::AbstractArray{T,N}, obsdim::ObsDim.Constant{M})
    StandardScaler(vec(mean(X, M)), vec(std(X, M)), obsdim)
end

function StandardScaler{T<:Real,M}(X::AbstractArray{T,M}, ::ObsDim.Last)
    StandardScaler(X, ObsDim.Constant{M})
end

function StandardScaler{T<:Real,M}(X::AbstractArray{T,M}, offset, scale, obsdim=LearnBase.default_obsdim(X))
    StandardScaler(offset, scale, convert(ObsDimension, obsdim))
end

function StandardScaler{T<:Real,N}(X::AbstractArray{T,N}, offset, scale, ::ObsDim.Last)
    StandardScaler(offset, scale, ObsDim.Constant{N})
end

function StandardScaler(D::AbstractDataFrame)
    flt_1 = Bool[T <: Real for T in eltypes(D)]
    flt_2 = Bool[any(isna(D[colname])) for colname in names(D)]
    flt = !(flt_1 | flt_2)
    offset = Float64[mean(D[colname]) for colname in names(D)[flt]]
    scale = Float64[std(D[colname]) for colname in names(D)[flt]]
    StandardScaler(offset, scale, ObsDim.Constant{1})
end

function StandardScaler(D::AbstractDataFrame, offset, scale)
    StandardScaler(offset, scale, ObsDim.Constant{1})
end

function StatsBase.fit{T<:Real}(::Type{StandardScaler}, X::AbstractMatrix{T}; obsdim=LearnBase.default_obsdim(X))
    StandardScaler(X, obsdim=obsdim)
end

function transform!{T<:AbstractFloat,N}(X::AbstractArray{T,N}, cs::StandardScaler)
    standardize!(X, cs.offset, cs.scale, obsdim=cs.obsdim)
    X
end

function transform{T<:AbstractFloat,N}(X::AbstractArray{T,N}, cs::StandardScaler)
    Xnew = copy(X)
    transform!(Xnew, cs)
end

function transform{T<:Real,N}(X::AbstractArray{T,N}, cs::StandardScaler)
    Xnew = convert(AbstractArray{Float64, N}, X)
    transform!(Xnew, cs)
    Xnew
end
