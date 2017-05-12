"""
    μ, σ = standardize!(X[, μ, σ, obsdim])

Center `X` along `obsdim` around the corresponding entry in the
vector `μ` and then standardize each feature using the corresponding
entry in the vector `σ`.
"""
function standardize!(X, μ, σ; obsdim=LearnBase.default_obsdim(X), operate_on=default_scalerange(X, convert(ObsDimension, obsdim)))
    standardize!(X, μ, σ, convert(ObsDimension, obsdim), operate_on)
end

function standardize!{T,N}(X::AbstractArray{T,N}, μ, σ, ::ObsDim.Last, operate_on)
    standardize!(X, μ, σ, ObsDim.Constant{N}(), operate_on)
end

function standardize!(X; obsdim=LearnBase.default_obsdim(X), operate_on=default_scalerange(X, convert(ObsDimension, obsdim)))
    standardize!(X, convert(ObsDimension, obsdim), operate_on)
end

function standardize!{T,N}(X::AbstractArray{T,N}, ::ObsDim.Last, operate_on)
    standardize!(X, ObsDim.Constant{N}(), operate_on)
end

function standardize!{T,N,M}(X::AbstractArray{T,N}, obsdim::ObsDim.Constant{M}, operate_on)
    μ = vec(mean(X, M))[operate_on]
    σ = vec(std(X, M))[operate_on]
    standardize!(X, μ, σ, obsdim, operate_on)
end

function standardize!{M}(X::AbstractVector, ::ObsDim.Constant{M}, operate_on)
    μ = mean(X)
    σ = std(X)
    for i in operate_on 
        X[i] = (X[i] - μ) / σ
    end
    μ, σ
end

function standardize!(X::AbstractMatrix, μ::AbstractVector, σ::AbstractVector, ::ObsDim.Constant{2}, operate_on)
    σ[σ .== 0] = 1
    nVars, nObs = size(X)
    for iObs in 1:nObs
        @inbounds for (i, iVar) in enumerate(operate_on)
            X[iVar, iObs] = (X[iVar, iObs] - μ[i]) / σ[i]
        end
    end
    μ, σ
end

function standardize!(X::AbstractMatrix, μ::AbstractVector, σ::AbstractVector, ::ObsDim.Constant{1}, operate_on)
    σ[σ .== 0] = 1
    nObs, nVars = size(X)
    for (i, iVar) in enumerate(operate_on)
        @inbounds for iObs in 1:nObs
            X[iObs, iVar] = (X[iObs, iVar] - μ[i]) / σ[i]
        end
    end
    μ, σ
end

function standardize!{M}(X::AbstractVector, μ::AbstractVector, σ::AbstractVector, ::ObsDim.Constant{M}, operate_on)
    @inbounds for (i, iVar) in enumerate(operate_on) 
        X[iVar] = (X[iVar] - μ[i]) / σ[i]
    end
    μ, σ
end

function standardize!{M}(X::AbstractVector, μ::AbstractFloat, σ::AbstractFloat, ::ObsDim.Constant{M}, operate_on)
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


immutable StandardScaler{T,U,M,I}
    offset::Vector{T}
    scale::Vector{U}
    obsdim::ObsDim.Constant{M}
    operate_on::Vector{I}
end

function StandardScaler{T<:Real,M}(X::AbstractArray{T,M}; obsdim=LearnBase.default_obsdim(X), operate_on=default_scalerange(X, convert(ObsDimension, obsdim)))
    StandardScaler(X, convert(ObsDimension, obsdim), operate_on)
end

function StandardScaler{T<:Real,M}(X::AbstractArray{T,M}, ::ObsDim.Last, operate_on)
    StandardScaler(X, ObsDim.Constant{M}(), operate_on)
end

function StandardScaler{T<:Real,N,M}(X::AbstractArray{T,N}, obsdim::ObsDim.Constant{M}, operate_on)
    StandardScaler(vec(mean(X, M))[operate_on], vec(std(X, M))[operate_on], obsdim, operate_on)
end

function StandardScaler{T<:Real,M}(X::AbstractArray{T,M}, offset, scale; obsdim=LearnBase.default_obsdim(X), operate_on=default_scalerange(X, convert(ObsDimension, obsdim)))
    StandardScaler(offset, scale, convert(ObsDimension, obsdim), operate_on)
end

function StandardScaler{T<:Real,N}(X::AbstractArray{T,N}, offset, scale, ::ObsDim.Last, operate_on)
    StandardScaler(offset, scale, ObsDim.Constant{N}(), operate_on)
end

function StandardScaler(D::AbstractDataFrame; operate_on=default_scalerange(D))
    offset = Float64[mean(D[colname]) for colname in operate_on]
    scale = Float64[std(D[colname]) for colname in operate_on]
    StandardScaler(offset, scale, ObsDim.Constant{1}(), operate_on)
end

function StandardScaler(D::AbstractDataFrame, offset, scale; operate_on=default_scalerange(D))
    StandardScaler(offset, scale, ObsDim.Constant{1}(), operate_on)
end

function StatsBase.fit{T<:Real}(::Type{StandardScaler}, X::AbstractMatrix{T}; obsdim=LearnBase.default_obsdim(X), operate_on=default_scalerange(X, convert(ObsDimension, obsdim)))
    StandardScaler(X, obsdim, operate_on)
end

function transform!{T<:AbstractFloat,N}(X::AbstractArray{T,N}, cs::StandardScaler)
    standardize!(X, cs.offset, cs.scale, cs.obsdim, cs.operate_on)
    X
end

function transform!(D::AbstractDataFrame, cs::StandardScaler)
    standardize!(D, cs.operate_on, cs.offset, cs.scale)
    D
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

function transform(D::AbstractDataFrame, cs::StandardScaler)
    Dnew = copy(D)
    transform!(Dnew, cs)
    Dnew
end
