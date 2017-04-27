"""
    μ = center!(X[, μ, obsdim])

or

    μ = center!(D[, colnames, μ])

where `X` is of type Matrix or Vector and `D` of type DataFrame.

Center `X` along `obsdim` around the corresponding entry in the
vector `μ`. If `μ` is not specified then it defaults to the
feature specific means.

For DataFrames, `obsdim` is obsolete and centering is done column wise.
Instead the vector `colnames` allows to specify which columns to center.
If `colnames` is not provided all columns of type T<:Real are centered.

Example:

    X = rand(4, 100)
    D = DataFrame(A=rand(10), B=collect(1:10), C=[string(x) for x in 1:10])

    μ = center!(X, obsdim=2)
    μ = center!(X, ObsDim.First())
    μ = center!(D)
    μ = center!(D, [:A, :B])

"""
function center!(X, μ; obsdim=LearnBase.default_obsdim(X))
    center!(X, μ, convert(ObsDimension, obsdim))
end

function center!(X; obsdim=LearnBase.default_obsdim(X))
    center!(X, convert(ObsDimension, obsdim))
end

function center!{T,N}(X::AbstractArray{T,N}, μ::AbstractVector, ::ObsDim.Last)
    center!(X, μ, ObsDim.Constant{N}())
end

function center!{T,N}(X::AbstractArray{T,N}, ::ObsDim.Last)
    center!(X, ObsDim.Constant{N}())
end

function center!{T,N,M}(X::AbstractArray{T,N}, obsdim::ObsDim.Constant{M})
    μ = vec(mean(X, M))
    center!(X, μ, obsdim)
end

function center!{T}(X::AbstractVector{T}, ::ObsDim.Constant{1})
    μ = mean(X)
    for i in 1:length(X)
        X[i] = X[i] - μ
    end
    μ
end

function center!(X::AbstractVector, μ::AbstractVector, ::ObsDim.Constant{1})
    @inbounds for i in 1:length(X)
        X[i] = X[i] - μ[i]
    end
    μ
end

function center!(X::AbstractVector, μ::AbstractFloat, ::ObsDim.Constant{1})
    @inbounds for i in 1:length(X)
        X[i] = X[i] - μ
    end
    μ
end

function center!(X::AbstractMatrix, μ::AbstractVector, ::ObsDim.Constant{1})
    nObs, nVars = size(X)
    for iVar in 1:nVars
        @inbounds for iObs in 1:nObs
            X[iObs, iVar] = X[iObs, iVar] - μ[iVar]
        end
    end
    μ
end

function center!(X::AbstractMatrix, μ::AbstractVector, ::ObsDim.Constant{2})
    nVars, nObs = size(X)
    for iObs in 1:nObs
        @inbounds for iVar in 1:nVars
            X[iVar, iObs] = X[iVar, iObs] - μ[iVar]
        end
    end
    μ
end

# --------------------------------------------------------------------

function center!(D::AbstractDataFrame)
    μ_vec = Float64[]

    flt = Bool[T <: Real for T in eltypes(D)]
    for colname in names(D)[flt]
        μ = mean(D[colname])
        center!(D, colname, μ)
        push!(μ_vec, μ)
    end
    μ_vec
end

function center!(D::AbstractDataFrame, colnames::AbstractVector{Symbol})
    μ_vec = Float64[]
    for colname in colnames
        if eltype(D[colname]) <: Real
            μ = mean(D[colname])
            if isna(μ)
                warn("Column \"$colname\" contains NA values, skipping rescaling of this column!")
                continue
            end
            center!(D, colname, μ)
            push!(μ_vec, μ)
        else
            warn("Skipping \"$colname\", centering only valid for columns of type T <: Real.")
        end
    end
    μ_vec
end

function center!(D::AbstractDataFrame, colnames::AbstractVector{Symbol}, μ::AbstractVector)
    for (icol, colname) in enumerate(colnames)
        if eltype(D[colname]) <: Real
            center!(D, colname, μ[icol])
        else
            warn("Skipping \"$colname\", centering only valid for columns of type T <: Real.")
        end
    end
    μ
end

function center!(D::AbstractDataFrame, colname::Symbol, μ)
    if sum(isna(D[colname])) > 0
        warn("Column \"$colname\" contains NA values, skipping centering on this column!")
    else
        newcol::Vector{Float64} = convert(Vector{Float64}, D[colname])
        nobs = length(newcol)
        @inbounds for i in eachindex(newcol)
            newcol[i] -= μ
        end
        D[colname] = newcol
    end
    μ
end
