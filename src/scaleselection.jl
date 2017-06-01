function default_scaleselection(X::AbstractMatrix, ::ObsDim.Constant{1})
    collect(1:size(X, 2))
end

function default_scaleselection(X::AbstractMatrix, ::ObsDim.Constant{2})
    collect(1:size(X, 1))
end

function default_scaleselection(X::AbstractMatrix, ::ObsDim.Last)
    collect(1:size(X, 1))
end

function default_scaleselection(x::AbstractVector)
    collect(1:length(x))
end

function default_scaleselection(x::AbstractVector, ::ObsDim.Last)
    collect(1:length(x))
end

function default_scaleselection{M}(x::AbstractVector, ::ObsDim.Constant{M})
    collect(1:length(x))
end

function default_scaleselection(D::AbstractDataFrame)
    flt1 = Bool[T <: Real for T in eltypes(D)]
    flt2 = Bool[any(isna(D[colname])) for colname in names(D)]
    flt = (flt1 & !flt2)
    names(D)[flt]
end

function valid_columns(D::AbstractDataFrame)
    valid_colnames = Symbol[]
    for colname in names(D)
        if (eltype(D[colname]) <: Real) & !(any(isna(D[colname])))
            push!(valid_colnames, colname)
        else
            warn("Skipping \"$colname\" because it either contains NA or is not of type <: Real")
        end
    end
    valid_colnames
end

function valid_columns(D::AbstractDataFrame, colnames)
    valid_colnames = Symbol[]
    for colname in colnames
        if (eltype(D[colname]) <: Real) & !(any(isna(D[colname])))
            push!(valid_colnames, colname)
        else
            warn("Skipping \"$colname\" because it either contains NA or is not of type <: Real")
        end
    end
    valid_colnames
end

