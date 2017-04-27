@testset "Test FeatureNormalizer model" begin
    e_x = collect(-5:.1:5)
    e_X = [e_x e_x.^2 e_x.^3]'

    cs = fit(FeatureNormalizer, e_X)
    @test vec(mean(e_X, 2)) ≈ cs.offset
    @test vec(std(e_X, 2)) ≈ cs.scale

    Xa = predict(cs, e_X)
    @test Xa != e_X
    @test abs(sum(mean(Xa, 2))) <= 10e-10
    @test std(Xa, 2) ≈ [1, 1, 1]
end
