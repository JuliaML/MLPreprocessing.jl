__precompile__()
module MLPreprocessing

using StatsBase
using LearnBase
using DataFrames

using LearnBase: ObsDimension, obs_dim, default_obsdim

export

    ObsDim,
    expand_poly,

    center!,
    rescale!

include("basis_expansion.jl")
include("center.jl")
include("rescale.jl")

end # module
