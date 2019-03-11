"""
    expand_poly(x, [degree = 5], [obsdim]) -> Matrix

Perform a simple polynomial basis expansion of the given `degree`
for the vector `x`.

The optional parameter `obsdim` can be used to specify which
dimension of the resulting `Matrix` should correspond to the
observations. By default each column will denote an observation,
which means that the resulting `Matrix` would have the size
`(degree, length(x))`.

```@jldoctest
julia> expand_poly(1:5, degree = 3)
3×5 Array{Int64,2}:
 1  2   3   4    5
 1  4   9  16   25
 1  8  27  64  125

julia> expand_poly(1f0:5f0, 3) # positional arguments are type-stable
3×5 Array{Float32,2}:
 1.0  2.0   3.0   4.0    5.0
 1.0  4.0   9.0  16.0   25.0
 1.0  8.0  27.0  64.0  125.0
```

Alternatively it is also possible to specify `obsdim = 1`, which
will cause each row to denote an observation.

```@jldoctest
julia> expand_poly(1:5, degree = 3, obsdim = 1)
5×3 Array{Int64,2}:
 1   1    1
 2   4    8
 3   9   27
 4  16   64
 5  25  125

julia> expand_poly(1:5, 3, ObsDim.First()); # same but type-stable
```
"""
function expand_poly(x::AbstractVector; degree::Integer = 5, obsdim = ObsDim.Last())
    expand_poly(x, degree, convert(LearnBase.ObsDimension, obsdim))
end

function expand_poly(x::AbstractVector, degree::Integer, ::ObsDim.Last)
    expand_poly(x, degree, ObsDim.Constant{2}())
end

function expand_poly(
    x::AbstractVector,
    degree::Integer,
    ::ObsDim.Constant{2} = ObsDim.Constant{2}()
)

    n = length(x)
    X = float.(zeros(T, (degree, n)))
    for i in 1:n
        for d in 1:degree
            @inbounds X[d, i] += float(x[i])^d
        end
    end
    return X
end

function expand_poly(
    x::AbstractVector,
    degree::Integer,
    ::ObsDim.Constant{1}
)

    n = length(x)
    X = float.(zeros(T, (degree, n)))
    for d in 1:degree
        for i in 1:n
            @inbounds X[i, d] += float(x[i])^d
        end
    end
    return X
end
