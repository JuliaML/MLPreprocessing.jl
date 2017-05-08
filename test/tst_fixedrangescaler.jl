R = reshape(1:40, 10, 4) 
F = convert(Matrix{Float64}, R)
r1 = collect(1:4) 
r2 = collect(1:10) 
@testset "Array" begin
    scaler = fit(FixedRangeScaler, F)
    X = transform(F, scaler)
    @test mean(X[:,end]) ≈ 1
    @test mean(X[:,1]) ≈ 0
    @test maximum(X) == 1
    @test minimum(X) == 0

    scaler = fit(FixedRangeScaler, F, obsdim=1)
    X = transform(F, scaler)
    @test mean(X[1,:]) ≈ 0
    @test mean(X[end,:]) ≈ 1
    @test maximum(X) == 1
    @test minimum(X) == 0

    scaler = fit(FixedRangeScaler, F, -2, 2)
    X = transform(F, scaler)
    @test mean(X[:,end]) ≈ 2
    @test mean(X[:,1]) ≈ -2
    @test maximum(X) == 2
    @test minimum(X) == -2 

    scaler = fit(FixedRangeScaler, F, -2, 2, obsdim=1)
    X = transform(F, scaler)
    @test mean(X[1,:]) ≈ -2 
    @test mean(X[end,:]) ≈ 2
    @test maximum(X) == 2
    @test minimum(X) == -2

    scaler = fit(FixedRangeScaler, R, -2, 2, obsdim=1)
    X = transform(R, scaler)
    @test mean(X[1,:]) ≈ -2 
    @test mean(X[end,:]) ≈ 2
    @test maximum(X) == 2
    @test minimum(X) == -2

    scaler = fit(FixedRangeScaler, R, -2, 2, obsdim=1)
    r = transform(r1, scaler)
    @test r == -[2, 6, 10, 14]

    scaler = fit(FixedRangeScaler, R, -2, 2, obsdim=2)
    r = transform(r2, scaler)
    @test r == -2 * ones(size(R, 1))
end
