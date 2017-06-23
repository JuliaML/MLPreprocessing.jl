"""
    μ, σ = standardize!(X[, μ, σ; obsdim, operate_on])

or

    μ, σ = standardize!(D[, μ, σ; operate_on])

Standardize `X` along `obsdim` according to X = (X - μ) / σ.
If μ and σ are omitted they are computed such that variables have a mean of zero



`μ`         :  Vector or value describing the translation.
               Defaults to mean(X, 2)

`σ`         :  Vector or value describing the scale.
               Defaults to std(X, 2)

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

    μ, σ = standardize!(X, obsdim=2)
    μ, σ = standardize!(X, ObsDim.First())
    μ, σ = standardize!(X, obsdim=1, operate_on=[1,3]
    μ, σ = standardize!(X, [7.0,8.0], [1,1], obsdim=1, operate_on=[1,3]
    μ, σ = standardize!(D)
    μ, σ = standardize!(D, operate_on=[:A,:B])
    μ, σ = standardize!(D, [-1,-1], [2,2], operate_on=[:A,:B])
"""
function standardize!(X, μ, σ; obsdim=LearnBase.default_obsdim(X), operate_on=default_scaleselection(X, convert(ObsDimension, obsdim)))
    standardize!(X, μ, σ, convert(ObsDimension, obsdim), operate_on)
end

function standardize!{T,N}(X::AbstractArray{T,N}, μ, σ, ::ObsDim.Last, operate_on)
    standardize!(X, μ, σ, ObsDim.Constant{N}(), operate_on)
end

function standardize!(X; obsdim=LearnBase.default_obsdim(X), operate_on=default_scaleselection(X, convert(ObsDimension, obsdim)))
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
function standardize!(D::AbstractDataFrame; operate_on=default_scaleselection(D))
    standardize!(D, operate_on)
end

function standardize!(D::AbstractDataFrame, colnames::AbstractVector{Symbol})
    μ_vec = Float64[]
    σ_vec = Float64[]

    for colname in colnames
        if eltype(D[colname]) <: Real
            μ = mean(D[colname])
            σ = std(D[colname])
            if isna(μ)
                warn("Skipping \"$colname\" because it contains NA values")
                continue
            end
            standardize!(D, μ, σ, colname)
            push!(μ_vec, μ)
            push!(σ_vec, σ)
        else
            warn("Skipping \"$colname\" because data is not of type T <: Real.")
        end
    end
    μ_vec, σ_vec
end

function standardize!(D::AbstractDataFrame, μ::AbstractVector, σ::AbstractVector; operate_on=default_scaleselection(D))
    standardize!(D, μ, σ, operate_on)
end

function standardize!(D::AbstractDataFrame, μ::AbstractVector, σ::AbstractVector, colnames::AbstractVector{Symbol})
    for (icol, colname) in enumerate(colnames)
        standardize!(D, μ[icol], σ[icol], colname)
    end
    μ, σ
end

function standardize!(D::AbstractDataFrame, μ::Real, σ::Real, colname::Symbol)
    if any(isna, D[colname]) | !(eltype(D[colname]) <: Real)
        warn("Skipping \"$colname\" because it contains NA values or is not of type <: Real")
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

"""
`StandardScaler` is used with the functions `fit()`, `transform()` and `fit_transform()`
to standardize data in a Matrix `X` or DataFrame according to Xnew = (X - μ) / σ.
After fitting a `StandardScaler` to one data set, it can be used to perform the same
transformation to a new set of data. E.g. fit the `StandardScaler` to your training
data and then apply the scaling to the test data at a later stage. (See examples below).

    fit(StandardScaler, X[, μ, σ; obsdim, operate_on])

    fit_transform(StandardScaler, X[, μ, σ; obsdim, operate_on])

`X`         :  Data of type Matrix or `DataFrame`.

`μ`         :  Vector or scalar describing the translation.
               Defaults to mean(X, obsdim)

`σ`         :  Vector or scalar describing the scale.
               Defaults to std(X, obsdim)

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
    x = rand(4)
    Dtrain = DataFrame(A=rand(10), B=collect(1:10), C=[string(x) for x in 1:10])
    Dtest = DataFrame(A=rand(10), B=collect(1:10), C=[string(x) for x in 1:10])

    scaler = fit(StandardScaler, Xtrain)
    scaler = fit(StandardScaler, Xtrain, obsdim=1)
    scaler = fit(StandardScaler, Xtrain, obsdim=1, operate_on=[1,3])
    transform(Xtest, scaler)
    transform!(Xtest, scaler)
    transform(x, scaler)
    transform!(x, scaler)

    scaler = fit(StandardScaler, Dtrain)
    scaler = fit(StandardScaler, Dtrain, operate_on=[:A,:B])
    transform(Dtest, scaler)
    transform!(Dtest, scaler)

    Xscaled, scaler = fit_transform(StandardScaler, X, obsdim=1, operate_on=[1,2,4])
    scaler = fit_transform!(StandardScaler, X, obsdim=1, operate_on=[1,2,4])

Note that for `transform!` the data matrix `X` has to be of type <: AbstractFloat
as the scaling occurs inplace. (E.g. cannot be of type Matrix{Int64}). This is not
the case for `transform` however.
For `DataFrames` `transform!` can be used on columns of type <: Integer.
"""
immutable StandardScaler{T<:Real,U<:Real,I,M}
    offset::Vector{T}
    scale::Vector{U}
    obsdim::ObsDim.Constant{M}
    operate_on::Vector{I}
end

function StandardScaler{T<:Real,M}(X::AbstractArray{T,M}; obsdim=LearnBase.default_obsdim(X), operate_on=default_scaleselection(X, convert(ObsDimension, obsdim)))
    StandardScaler(X, convert(ObsDimension, obsdim), operate_on)
end

function StandardScaler{T<:Real,M}(X::AbstractArray{T,M}, ::ObsDim.Last, operate_on)
    StandardScaler(X, ObsDim.Constant{M}(), operate_on)
end

function StandardScaler{T<:Real,N,M}(X::AbstractArray{T,N}, obsdim::ObsDim.Constant{M}, operate_on::AbstractVector)
    offset = vec(mean(X,M))[operate_on] 
    scale = vec(std(X, M))[operate_on]
    StandardScaler(offset, scale, obsdim, operate_on)
end

function StandardScaler(D::AbstractDataFrame; operate_on=default_scaleselection(D))
    StandardScaler(D, operate_on)
end

function StandardScaler(D::AbstractDataFrame, operate_on::Vector{Symbol})
    colnames = valid_columns(D, operate_on)
    offset = Float64[mean(D[colname]) for colname in colnames]
    scale = Float64[std(D[colname]) for colname in colnames]
    StandardScaler(offset, scale, ObsDim.Constant{1}(), colnames)
end

function StandardScaler(D::AbstractDataFrame, offset, scale; operate_on=default_scaleselection(D))
    colnames = valid_columns(D)
    StandardScaler(offset, scale, ObsDim.Constant{1}(), colnames)
end

function StatsBase.fit{T<:Real}(::Type{StandardScaler}, X::AbstractMatrix{T}; obsdim=LearnBase.default_obsdim(X), operate_on=default_scaleselection(X, convert(ObsDimension, obsdim)))
    StandardScaler(X, convert(ObsDimension, obsdim), operate_on)
end

function fit_transform{T<:Real}(::Type{StandardScaler}, X::AbstractMatrix{T}; obsdim=LearnBase.default_obsdim(X), operate_on=default_scaleselection(X, convert(ObsDimension, obsdim)))
    scaler = StandardScaler(X, convert(ObsDimension, obsdim), operate_on)
    Xnew = transform(X, scaler)
    return Xnew, scaler
end

function fit_transform!{T<:Real}(::Type{StandardScaler}, X::AbstractMatrix{T}; obsdim=LearnBase.default_obsdim(X), operate_on=default_scaleselection(X, convert(ObsDimension, obsdim)))
    scaler = StandardScaler(X, convert(ObsDimension, obsdim), operate_on)
    transform!(X, scaler)
    return scaler
end

function StatsBase.fit(::Type{StandardScaler}, D::AbstractDataFrame; operate_on=default_scaleselection(D))
    StandardScaler(D, operate_on)
end

function fit_transform(::Type{StandardScaler}, D::AbstractDataFrame; operate_on=default_scaleselection(D))
    scaler = StandardScaler(D, operate_on)
    Dnew = transform(D, scaler)
    return Dnew, scaler
end

function fit_transform!(::Type{StandardScaler}, D::AbstractDataFrame; operate_on=default_scaleselection(D))
    scaler = StandardScaler(D, operate_on)
    transform!(D, scaler)
    return scaler
end

function transform!{T<:AbstractFloat,N}(X::AbstractArray{T,N}, cs::StandardScaler)
    standardize!(X, cs.offset, cs.scale, cs.obsdim, cs.operate_on)
    X
end

function transform!(D::AbstractDataFrame, cs::StandardScaler)
    standardize!(D, cs.offset, cs.scale, cs.operate_on)
    D
end

function transform{T<:AbstractFloat,N}(X::AbstractArray{T,N}, cs::StandardScaler)
    Xnew = deepcopy(X)
    transform!(Xnew, cs)
end

function transform{T<:Real,N}(X::AbstractArray{T,N}, cs::StandardScaler)
    Xnew = convert(AbstractArray{Float64, N}, X)
    transform!(Xnew, cs)
    Xnew
end

function transform(D::AbstractDataFrame, cs::StandardScaler)
    Dnew = deepcopy(D)
    transform!(Dnew, cs)
    Dnew
end
