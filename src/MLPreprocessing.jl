__precompile__()
module MLPreprocessing

using Statistics
using StatsBase
using LearnBase
using DataFrames

using LearnBase: ObsDimension, default_obsdim

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
    transform!

include("scaleselection.jl")
include("basis_expansion.jl")
include("center.jl")
include("standardize.jl")
include("fixedrange.jl")

end # module
