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
    standardize!,
    fixedrange!,

    StandardScaler,
    FixedRangeScaler,
    fit,
    fit_transform,
    fit_transform!,
    transform,
    transform!,

    onehot!

 
include("scaleselection.jl")
include("basis_expansion.jl")
include("center.jl")
include("standardize.jl")
include("fixedrange.jl")
include("encoding.jl")

end # module
