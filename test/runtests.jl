using MLPreprocessing
using DataFrames
using Base.Test

tests = [
    "tst_expand.jl"
    "tst_center.jl"
    "tst_rescale.jl"
    "tst_featurenormalizer.jl"
]

for t in tests
    @testset "$t" begin
        include(t)
    end
end
