"""
    center!(X::AbstractMatrix, [μ::AbstractVector]; [obsdim], [operate_on]) -> μ

Shift the values in `X` along `obsdim` by the corresponding
values in `μ`. This means it shifts each feature across all
observations. The optional parameter `operate_on` can be used to
limit the operation to specific "features".

# Arguments

- `μ`: Vector describing the translation. Defaults to `mean(X, 2)`.

- `obsdim`: Specify which axis corresponds to observations.
  Defaults to `obsdim = 2`, which implies that each column of the
  matrix is an observation.

- `operate_on`: Specify the indices of "features", which
  depending on `pobsdim` are either the columns or rows, to be
  centered. Defaults to all columns/rows. For example
  `operate_on = [1,3]` will perform centering on the rows with
  index 1 and 3 (if `obsdim = 2`).

# Examples

```@jldoctests
julia> using MLPreprocessing

julia> X = [1. 2. 3.; 101. 103. 105.]
2×3 Array{Float64,2}:
   1.0    2.0    3.0
 101.0  103.0  105.0

julia> mu = center!(X)
2-element Array{Float64,1}:
   2.0
 103.0

julia> X
2×3 Array{Float64,2}:
 -1.0  0.0  1.0
 -2.0  0.0  2.0

julia> x = [2., 102.]
2-element Array{Float64,1}:
   2.0
 102.0

julia> center!(x, mu); x
2-element Array{Float64,1}:
  0.0
 -1.0

julia> X = [1. 2. 3.; 101. 103. 105.]'
3×2 Array{Float64,2}:
 1.0  101.0
 2.0  103.0
 3.0  105.0

julia> mu = center!(X, obsdim=1, operate_on=[2])
1-element Array{Float64,1}:
 103.0

julia> X
3×2 Array{Float64,2}:
 1.0  -2.0
 2.0   0.0
 3.0   2.0
```
"""
function center!(X::AbstractMatrix;
                 obsdim = LearnBase.default_obsdim(X),
                 operate_on = default_scaleselection(X, convert(ObsDimension, obsdim)))
    center!(X, convert(ObsDimension, obsdim), operate_on)
end

function center!(X::AbstractMatrix,
                 obsdim::ObsDimension;
                 operate_on = default_scaleselection(X, obsdim))
    center!(X, obsdim, operate_on)
end

function center!(X::AbstractMatrix,
                 obsdim::ObsDim.Last,
                 operate_on::AbstractVector)
    center!(X, ObsDim.Constant{2}(), operate_on)
end

function center!(X::AbstractMatrix,
                 obsdim::ObsDim.Constant{M},
                 operate_on::AbstractVector
                ) where {M}
    μ = vec(mean(X, M))[operate_on]
    center!(X, μ, obsdim, operate_on)
end

function center!(X::AbstractMatrix,
                 μ::AbstractVector;
                 obsdim = LearnBase.default_obsdim(X),
                 operate_on = default_scaleselection(X, convert(ObsDimension, obsdim)))
    center!(X, μ, convert(ObsDimension, obsdim), operate_on)
end

function center!(X::AbstractMatrix,
                 μ::AbstractVector,
                 obsdim::ObsDimension;
                 operate_on = default_scaleselection(X, obsdim))
    center!(X, μ, obsdim, operate_on)
end

function center!(X::AbstractMatrix,
                 μ::AbstractVector,
                 obsdim::ObsDim.Last,
                 operate_on)
    center!(X, μ, ObsDim.Constant{2}(), operate_on)
end

function center!(X::AbstractMatrix,
                 μ::AbstractVector,
                 ::ObsDim.Constant{1},
                 operate_on)
    @assert length(μ) == length(operate_on)
    nObs, nVars = size(X)
    for (i, iVar) in enumerate(operate_on)
        for iObs in 1:nObs
            X[iObs, iVar] = X[iObs, iVar] - μ[i]
        end
    end
    μ
end

function center!(X::AbstractMatrix,
                 μ::AbstractVector,
                 ::ObsDim.Constant{2},
                 operate_on)
    @assert length(μ) == length(operate_on)
    nVars, nObs = size(X)
    for iObs in 1:nObs
        for (i, iVar) in enumerate(operate_on)
            X[iVar, iObs] = X[iVar, iObs] - μ[i]
        end
    end
    μ
end

# --------------------------------------------------------------------

function center!(x::AbstractVector, μ = mean(x); operate_on = default_scaleselection(x))
    center!(x, μ, operate_on)
end

function center!(x::AbstractVector,
                 μ::Number,
                 operate_on::AbstractVector)
    for iVar in operate_on
        x[iVar] = x[iVar] - μ
    end
    μ
end

function center!(x::AbstractVector,
                 μ::AbstractVector,
                 operate_on::AbstractVector)
    @assert length(μ) == length(operate_on)
    for (i, iVar) in enumerate(operate_on)
        x[iVar] = x[iVar] - μ[i]
    end
    μ
end

# --------------------------------------------------------------------

"""
    center!(df::AbstractDataFrame, [μ]; [operate_on]) -> μ

Shift the values in each column of `df` by the corresponding
values in `μ`.

This means it shifts each feature across all observations. The
optional parameter `operate_on` can be used to limit the
operation to specific columns of the data frame. Note that
columns containing missing or non-numeric values are skipped.

# Arguments

- `μ`: Vector or value describing the translation. Defaults to
  `mean(X, 2)`

- `operate_on`: Vector of symbols that specify the indices of
  columns to be centered. Defaults to all columns.

# Examples

```@jldoctest
julia> using MLPreprocessing, DataFrames

julia> df = DataFrame(A=collect(5:-1:1).*0.1, B=collect(1:5), C=[string("foo", x) for x in 1:5])
5×3 DataFrames.DataFrame
│ Row │ A   │ B │ C    │
├─────┼─────┼───┼──────┤
│ 1   │ 0.5 │ 1 │ foo1 │
│ 2   │ 0.4 │ 2 │ foo2 │
│ 3   │ 0.3 │ 3 │ foo3 │
│ 4   │ 0.2 │ 4 │ foo4 │
│ 5   │ 0.1 │ 5 │ foo5 │

julia> μ = center!(df, operate_on = [:A, :B])
2-element Array{Float64,1}:
 0.3
 3.0

julia> df
5×3 DataFrames.DataFrame
│ Row │ A    │ B    │ C    │
├─────┼──────┼──────┼──────┤
│ 1   │ 0.2  │ -2.0 │ foo1 │
│ 2   │ 0.1  │ -1.0 │ foo2 │
│ 3   │ 0.0  │ 0.0  │ foo3 │
│ 4   │ -0.1 │ 1.0  │ foo4 │
│ 5   │ -0.2 │ 2.0  │ foo5 │
```
"""
function center!(D::AbstractDataFrame; operate_on=default_scaleselection(D))
    center!(D, operate_on)
end

function center!(D::AbstractDataFrame, operate_on::AbstractVector{Symbol})
    μ_vec = Float64[]
    for colname in operate_on
        if eltype(D[colname]) <: Real
            μ = mean(D[colname])
            if ismissing(μ)
                warn("Skipping \"$colname\" because it contains missing values")
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

function center!(D::AbstractDataFrame, μ::AbstractVector; operate_on=default_scaleselection(D))
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
    if sum([ismissing(value) for value in D[colname]]) > 0
        warn("Skipping \"$colname\" because it contains missing values")
    else
        newcol::Vector{Float64} = convert(Vector{Float64}, D[colname])
        center!(newcol, μ)
        D[colname] = newcol
    end
    μ
end
