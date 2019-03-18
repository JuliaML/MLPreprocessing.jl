function default_scaleselection(X::AbstractMatrix, ::ObsDim.Constant{1})
    1:size(X, 2)
end

function default_scaleselection(X::AbstractMatrix, ::Union{ObsDim.Constant{2},ObsDim.Last})
    1:size(X, 1)
end

function default_scaleselection(x::AbstractVector)
    1:length(x)
end

function default_scaleselection(x::AbstractVector, ::ObsDim.Last)
    1:length(x)
end

function default_scaleselection(x::AbstractVector, ::ObsDim.Constant{M}) where {M}
    collect(1:length(x))
end

function default_scaleselection(D::AbstractDataFrame)
    valid_columns(D)
end

function valid_columns(D::AbstractDataFrame)
    valid_colnames = Symbol[]
    for colname in names(D)
        if (eltype(D[colname]) <: Real) & !any(ismissing, D[colname])
            push!(valid_colnames, colname)
        else
            warn("Skipping \"$colname\" because it either contains missing values or is not of type <: Real")
        end
    end
    valid_colnames
end

function valid_columns(D::AbstractDataFrame, colnames)
    valid_colnames = Symbol[]
    for colname in colnames
        if (eltype(D[colname]) <: Real) & !(any(ismissing, D[colname]))
            push!(valid_colnames, colname)
        else
            warn("Skipping \"$colname\" because it either contains missing values or is not of type <: Real")
        end
    end
    valid_colnames
end

