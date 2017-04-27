"""
    expand_poly(x, [degree = 5], [obsdim]) -> Matrix

Perform a polynomial basis expansion of the given `degree` for
the vector `x`.

The optional parameter `obsdim` can be used to specify which
dimension of the resulting `Matrix` should denote the
observations. By default the each column will denote an
observation, which means that the returning `Matrix` will have
the size `(degree, length(x))`.

```julia
julia> expand_poly(1:5, degree = 3)
3×5 Array{Float64,2}:
 1.0  2.0   3.0   4.0    5.0
 1.0  4.0   9.0  16.0   25.0
 1.0  8.0  27.0  64.0  125.0

julia> expand_poly(1:5, 3); # same but type-stable
```

If you want each row to denote an observation you can set
`obsdim = 1`.

```julia
julia> expand_poly(1:5, degree = 3, obsdim = 1)
5×3 Array{Float64,2}:
 1.0   1.0    1.0
 2.0   4.0    8.0
 3.0   9.0   27.0
 4.0  16.0   64.0
 5.0  25.0  125.0

julia> expand_poly(1:5, 3, ObsDim.First()); # same but type-stable
```
"""
function expand_poly(x::AbstractVector; degree::Integer = 5, obsdim = ObsDim.Last())
    expand_poly(x, degree, convert(LearnBase.ObsDimension, obsdim))
end

function expand_poly(x::AbstractVector, degree::Integer, ::ObsDim.Last)
    expand_poly(x, degree, ObsDim.Constant{2}())
end

function expand_poly{T<:Number}(x::AbstractVector{T}, degree::Integer, ::ObsDim.Constant{2} = ObsDim.Constant{2}())
    n = length(x)
    # FIXME: consider support for colorants (i.e. don't hardcode Float64)
    x_vec = convert(Vector{Float64}, x)
    X = zeros(Float64, (degree, n))
    @inbounds for i = 1:n
        for d = 1:degree
            X[d, i] += x_vec[i]^(d)
        end
    end
    X
end

function expand_poly{T<:Number}(x::AbstractVector{T}, degree::Integer, ::ObsDim.Constant{1})
    n = length(x)
    # FIXME: consider support for colorants (i.e. don't hardcode Float64)
    x_vec = convert(Vector{Float64}, x)
    X = zeros(Float64, (n, degree))
    @inbounds for d = 1:degree
        for i = 1:n
            X[i, d] += x_vec[i]^(d)
        end
    end
    X
end
