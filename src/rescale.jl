"""
    μ, σ = rescale!(X[, μ, σ, obsdim])

or

    μ, σ = rescale!(D[, colnames, μ, σ])

where `X` is of type Matrix or Vector and `D` of type DataFrame.

Center `X` along `obsdim` around the corresponding entry in the
vector `μ` and then rescale each feature using the corresponding
entry in the vector `σ`.

For DataFrames, `obsdim` is obsolete and centering is done column wise.
The vector `colnames` allows to specify which columns to center.
If `colnames` is not provided all columns of type T<:Real are centered.

Example:

    X = rand(4, 100)
    D = DataFrame(A=rand(10), B=collect(1:10), C=[string(x) for x in 1:10])

    μ, σ = rescale!(X, obsdim=2)
    μ, σ = rescale!(X, ObsDim.First())
    μ, σ = rescale!(D)
    μ, σ = rescale!(D, [:A, :B])

"""
function rescale!(X, μ, σ; obsdim=LearnBase.default_obsdim(X))
    rescale!(X, μ, σ, convert(ObsDimension, obsdim))
end

function rescale!{T,N}(X::AbstractArray{T,N}, μ, σ, ::ObsDim.Last)
    rescale!(X, μ, σ, ObsDim.Constant{N}())
end

function rescale!(X; obsdim=LearnBase.default_obsdim(X))
    rescale!(X, convert(ObsDimension, obsdim))
end

function rescale!{T,N}(X::AbstractArray{T,N}, ::ObsDim.Last)
    rescale!(X, ObsDim.Constant{N}())
end

function rescale!{T,N,M}(X::AbstractArray{T,N}, obsdim::ObsDim.Constant{M})
    μ = vec(mean(X, M))
    σ = vec(std(X, M))
    rescale!(X, μ, σ, obsdim)
end

function rescale!(X::AbstractVector, ::ObsDim.Constant{1})
    μ = mean(X)
    σ = std(X)
    @inbounds for i in 1:length(X)
        X[i] = (X[i] - μ) / σ
    end
    μ, σ
end

function rescale!(X::AbstractMatrix, μ::AbstractVector, σ::AbstractVector, ::ObsDim.Constant{2})
    σ[σ .== 0] = 1
    nVars, nObs = size(X)
    for iObs in 1:nObs
        @inbounds for iVar in 1:nVars
            X[iVar, iObs] = (X[iVar, iObs] - μ[iVar]) / σ[iVar]
        end
    end
    μ, σ
end

function rescale!(X::AbstractMatrix, μ::AbstractVector, σ::AbstractVector, ::ObsDim.Constant{1})
    σ[σ .== 0] = 1
    nObs, nVars = size(X)
    for iVar in 1:nVars
        @inbounds for iObs in 1:nObs
            X[iObs, iVar] = (X[iObs, iVar] - μ[iVar]) / σ[iVar]
        end
    end
    μ, σ
end

function rescale!(X::AbstractVector, μ::AbstractVector, σ::AbstractVector, ::ObsDim.Constant{1})
    @inbounds for i in 1:length(X)
        X[i] = (X[i] - μ[i]) / σ[i]
    end
    μ, σ
end

function rescale!(X::AbstractVector, μ::AbstractFloat, σ::AbstractFloat, ::ObsDim.Constant{1})
    @inbounds for i in 1:length(X)
        X[i] = (X[i] - μ) / σ
    end
    μ, σ
end

# --------------------------------------------------------------------

function rescale!(D::AbstractDataFrame)
    μ_vec = Float64[]
    σ_vec = Float64[]

    flt = Bool[T <: Real for T in eltypes(D)]
    for colname in names(D)[flt]
        μ = mean(D[colname])
        σ = std(D[colname])
        rescale!(D, colname, μ, σ)
        push!(μ_vec, μ)
        push!(σ_vec, σ)
    end
    μ_vec, σ_vec
end

function rescale!(D::AbstractDataFrame, colnames::Vector{Symbol})
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
            rescale!(D, colname, μ, σ)
            push!(μ_vec, μ)
            push!(σ_vec, σ)
        else
            warn("Skipping \"$colname\", rescaling only valid for columns of type T <: Real.")
        end
    end
    μ_vec, σ_vec
end

function rescale!(D::AbstractDataFrame, colnames::Vector{Symbol}, μ::AbstractVector, σ::AbstractVector)
    for (icol, colname) in enumerate(colnames)
        if eltype(D[colname]) <: Real
            rescale!(D, colname, μ[icol], σ[icol])
        else
            warn("Skipping \"$colname\", rescaling only valid for columns of type T <: Real.")
        end
    end
    μ, σ
end

function rescale!(D::AbstractDataFrame, colname::Symbol, μ, σ)
    if sum(isna(D[colname])) > 0
        warn("Column \"$colname\" contains NA values, skipping rescaling of this column!")
    else
        σ_div = σ == 0 ? one(σ) : σ
        newcol::Vector{Float64} = convert(Vector{Float64}, D[colname])
        nobs = length(newcol)
        @inbounds for i in eachindex(newcol)
            newcol[i] = (newcol[i] - μ) / σ_div
        end
        D[colname] = newcol
    end
    μ, σ
end
