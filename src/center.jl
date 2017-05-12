"""
    μ = center!(X[, μ; obsdim, operate_on])

or

    μ = center!(D[, μ; operate_on])

where `X` is of type Matrix or Vector and `D` of type DataFrame.

Shift `X` along `obsdim` by `μ` according to X = X - μ


`μ`         :  Vector or value describing the translation.
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

function center!(X, μ; obsdim=LearnBase.default_obsdim(X), operate_on=default_scalerange(X, convert(ObsDimension, obsdim)))
    center!(X, μ, convert(ObsDimension, obsdim), operate_on)
end

function center!(X; obsdim=LearnBase.default_obsdim(X), operate_on=default_scalerange(X, convert(ObsDimension, obsdim)))
    center!(X, convert(ObsDimension, obsdim), operate_on)
end

function center!{T,N,M}(X::AbstractArray{T,N}, obsdim::ObsDim.Constant{M}; operate_on=default_scalerange(X, convert(ObsDimension, obsdim)))
    center!(X, ObsDim.Constant{M}(), operate_on)
end

function center!{T,N}(X::AbstractArray{T,N}, obsdim::ObsDim.Last; operate_on=default_scalerange(X, convert(ObsDimension, obsdim)))
    center!(X, ObsDim.Constant{N}(), operate_on)
end

function center!{T,N,M}(X::AbstractArray{T,N}, obsdim::ObsDim.Constant{M}, operate_on::AbstractVector)
    center!(X, ObsDim.Constant{M}(), operate_on)
end

function center!{T,N}(X::AbstractArray{T,N}, obsdim::ObsDim.Last, operate_on::AbstractVector)
    center!(X, ObsDim.Constant{N}(), operate_on)
end

function center!{T,N,M}(X::AbstractArray{T,N}, obsdim::ObsDim.Constant{M}, operate_on::AbstractVector)
    μ = vec(mean(X, M))[operate_on]
    center!(X, μ, obsdim, operate_on)
end

function center!{T,N,M}(X::AbstractArray{T,N}, μ::AbstractVector, obsdim::ObsDim.Constant{M}; operate_on=default_scalerange(X, convert(ObsDimension, obsdim)))
    center!(X, μ, ObsDim.Constant{M}(), operate_on)
end

function center!{T,N}(X::AbstractArray{T,N}, μ::AbstractVector, obsdim::ObsDim.Last; operate_on=default_scalerange(X, convert(ObsDimension, obsdim)))
    center!(X, μ, ObsDim.Constant{N}(), operate_on)
end

function center!(X::AbstractMatrix, μ::AbstractVector, ::ObsDim.Constant{1}, operate_on)
    @assert length(μ) == length(operate_on)
    nObs, nVars = size(X)
    for (i, iVar) in enumerate(operate_on)
        @inbounds for iObs in 1:nObs
            X[iObs, iVar] = X[iObs, iVar] - μ[i]
        end
    end
    μ
end

function center!(X::AbstractMatrix, μ::AbstractVector, ::ObsDim.Constant{2}, operate_on)
    @assert length(μ) == length(operate_on)
    nVars, nObs = size(X)
    for iObs in 1:nObs
        @inbounds for (i, iVar) in enumerate(operate_on)
            X[iVar, iObs] = X[iVar, iObs] - μ[i]
        end
    end
    μ
end


function center!(x::AbstractVector; obsdim=LearnBase.default_obsdim(x), operate_on=default_scalerange(x))
    center!(x, convert(ObsDimension, obsdim), operate_on)
end

function center!{T,M}(x::AbstractVector{T}, ::ObsDim.Constant{M}, operate_on::AbstractVector)
    μ = mean(x)
    for iVar in operate_on
        x[iVar] = x[iVar] - μ
    end
    μ
end

function center!(x::AbstractVector, μ::AbstractVector, ::ObsDim.Constant{1}, operate_on::AbstractVector)
    @assert length(μ) == length(operate_on)
    @inbounds for (i, iVar) in enumerate(operate_on)
        x[iVar] = x[iVar] - μ[i]
    end
    μ
end

function center!(x::AbstractVector, μ::AbstractVector, ::ObsDim.Last, operate_on::AbstractVector)
    center!(x, μ, ObsDim.Constant{1}(), operate_on)
end

function center!(x::AbstractVector, μ::Real, ::ObsDim.Constant{1}, operate_on)
    @inbounds for i in operate_on 
        x[i] = x[i] - μ
    end
    μ
end

function center!(x::AbstractVector, μ::Real, ::ObsDim.Last, operate_on)
    center!(x, μ, ObsDim.Constant{1}(), operate_on)
end

# --------------------------------------------------------------------

function center!(D::AbstractDataFrame; operate_on=default_scalerange(D))
    center!(D, operate_on)
end

function center!(D::AbstractDataFrame, operate_on::AbstractVector{Symbol})
    μ_vec = Float64[]
    for colname in operate_on 
        if eltype(D[colname]) <: Real
            μ = mean(D[colname])
            if isna(μ)
                warn("Skipping \"$colname\" because it contains NA values")
                continue
            end
            center!(D, μ, colname)
            push!(μ_vec, μ)
        else
            warn("Skipping \"$colname\" because data is not of type T <: Real.")
        end
    end
    μ_vec
end

function center!(D::AbstractDataFrame, μ::AbstractVector; operate_on=default_scalerange(D))
    center!(D, μ, operate_on)
end

function center!(D::AbstractDataFrame, μ::AbstractVector, operate_on::AbstractVector{Symbol})
    for (icol, colname) in enumerate(operate_on)
        if eltype(D[colname]) <: Real
            center!(D, μ[icol], colname)
        else
            warn("Skipping \"$colname\" because data is not of type T <: Real.")
        end
    end
    μ
end

function center!(D::AbstractDataFrame, μ::Real, colname::Symbol)
    if sum(isna(D[colname])) > 0
        warn("Skipping \"$colname\" because it contains NA values")
    else
        newcol::Vector{Float64} = convert(Vector{Float64}, D[colname])
        center!(newcol, μ)
        D[colname] = newcol
    end
    μ
end
