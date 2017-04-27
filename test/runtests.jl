using MLPreprocessing
using Base.Test

tests = [
    "tst_expand.jl"
]

for t in tests
    @testset "$t" begin
        include(t)
    end
end
