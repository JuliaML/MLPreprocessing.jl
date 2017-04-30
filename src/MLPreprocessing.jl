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
    transform,
    transform!

    #= rescale!, =#
    #= FeatureNormalizer, =#
    #= predict, =#
    #= predict! =#

include("basis_expansion.jl")
include("center.jl")
include("rescale.jl")
include("featurenormalizer.jl")
include("standardize.jl")
include("fixedrange.jl")

end # module
