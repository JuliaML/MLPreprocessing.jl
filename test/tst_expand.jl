@testset "expand_poly" begin
    xi = 1:6
    xf = 1f0:6f0
    @test_throws MethodError expand_poly(xi, obsdim = 3)
    @test_throws MethodError expand_poly(xf, 5, ObsDim.Constant{3}())

    @testset "ObsDim.Last()" begin
        for x in [xi, xf]
            X = expand_poly(x, degree = 5)
            @test eltype(x) == eltype(X)
            @test @inferred(expand_poly(x)) == X
            @test @inferred(expand_poly(x, 5)) == X
            @test @inferred(expand_poly(x, 5, ObsDim.Last())) == X
            @test @inferred(expand_poly(x, 5, ObsDim.Constant{2}())) == X
            @test size(X) == (5, 6)
            for i in 1:6, k in 1:5
                @test X[k, i] === x[i]^k
            end
        end
    end

    @testset "ObsDim.First()" begin
        for x in [xi, xf]
            X = expand_poly(x, degree = 5, obsdim = 1)
            @test eltype(x) == eltype(X)
            @test expand_poly(x, obsdim = 1) == X
            @test @inferred(expand_poly(x, 5, ObsDim.First())) == X
            @test size(X) == (6, 5)
            for k in 1:5, i in 1:6
                @test X[i, k] === x[i]^k
            end
        end
    end

    @testset "Missing" begin
        xm = [1., 2., missing, 4., 5., 6.]
        @test_throws MethodError expand_poly(xm, obsdim = 3)
        @test_throws MethodError expand_poly(xm, 5, ObsDim.Constant{3}())
        X = @inferred(expand_poly(xm, 5))
        @test eltype(xm) == eltype(X)
        @test size(X) == (5, 6)
        for i in 1:6, k in 1:5
            if i != 3
                @test X[k, i] === xm[i]^k
            else
                @test ismissing(X[k, i])
            end
        end
    end
end
