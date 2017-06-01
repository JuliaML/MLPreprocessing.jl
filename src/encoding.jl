immutable OneHotEncoder{S}
    operate_on::Vector{S}
end

function onehot!(D::DataFrame, varname::Symbol)
    for keyword in unique(D[:, varname])
        sym_keyword = Symbol(string(varname) * "_" * keyword)
        D[sym_keyword] = zeros(Int, size(D, 1))
        for i in 1:size(D, 1)
            if D[i, varname] == keyword
                D[i, sym_keyword] = 1
            end
        end
    end
end

function onehot!(D::DataFrame, operate_on::Vector{Symbol})
    for varname in operate_on
        onehot!(D, varname)
    end
end

function onehot!(D::DataFrame; operate_on=default_categoricalselection(D))
    onehot!(D, operate_on)
end

