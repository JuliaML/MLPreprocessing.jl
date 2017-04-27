@testset "expand_poly" begin
    x = 1:6
    @test_throws MethodError expand_poly(x, obsdim = 3)
    @test_throws MethodError expand_poly(x, 5, ObsDim.Constant{3}())

    @testset "ObsDim.Last()" begin
        X = expand_poly(x, degree = 5)
        @test @inferred(expand_poly(x)) ≈ X
        @test @inferred(expand_poly(x, 5)) ≈ X
        @test @inferred(expand_poly(x, 5, ObsDim.Last())) ≈ X
        @test @inferred(expand_poly(x, 5, ObsDim.Constant{2}())) ≈ X

        @test size(X) == (5, 6)
        for i in 1:6, k in 1:5
            @test X[k, i] === Float64(x[i]^k)
        end
    end

    @testset "ObsDim.First()" begin
        X = expand_poly(x, degree = 5, obsdim = 1)
        @test expand_poly(x, obsdim = 1) ≈ X
        @test @inferred(expand_poly(x, 5, ObsDim.First())) ≈ X

        @test size(X) == (6, 5)
        for k in 1:5, i in 1:6
            @test X[i, k] === Float64(x[i]^k)
        end
    end
end
