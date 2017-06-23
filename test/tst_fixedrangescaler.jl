X = collect(Float64, reshape(1:40, 10, 4))
x = rand(10) * 10

D = DataFrame(A=rand(10), B=collect(1:10), C=[hex(x) for x in 11:20])
D_NA = deepcopy(D)
D_NA[1, :A] = NA
@testset "Array" begin
    scaler = fit(FixedRangeScaler, X)
    XX = transform(X, scaler)
    @test mean(XX[:,end]) ≈ 1
    @test mean(XX[:,1]) ≈ 0
    @test maximum(XX) == 1
    @test minimum(XX) == 0

    scaler = fit(FixedRangeScaler, X, obsdim=1)
    XX = transform(X, scaler)
    @test mean(XX[1,:]) ≈ 0
    @test mean(XX[end,:]) ≈ 1
    @test maximum(XX) == 1
    @test minimum(XX) == 0

    scaler = fit(FixedRangeScaler, X, -2, 2)
    XX = transform(X, scaler)
    @test mean(XX[:,end]) ≈ 2
    @test mean(XX[:,1]) ≈ -2
    @test maximum(XX) == 2
    @test minimum(XX) == -2 

    scaler = fit(FixedRangeScaler, X, -2, 2, obsdim=1)
    XX = transform(X, scaler)
    @test mean(XX[1,:]) ≈ -2 
    @test mean(XX[end,:]) ≈ 2
    @test maximum(XX) == 2
    @test minimum(XX) == -2

    scaler = fit(FixedRangeScaler, X, -2, 2, obsdim=2)
    XX = transform(X, scaler)
    @test mean(minimum(XX, 2)) ≈ -2 
    @test mean(maximum(XX, 2)) ≈ 2 
    @test maximum(XX) == 2
    @test minimum(XX) == -2

    scaler = fit(FixedRangeScaler, X, -2, 2, obsdim=1, operate_on=[1,2])
    XX = transform(X, scaler)
    @test mean(minimum(XX[:,[1,2]], 1)) ≈ -2 
    @test mean(maximum(XX[:,[1,2]], 1)) ≈ 2 

    scaler = fit(FixedRangeScaler, X, -2, 2, obsdim=2, operate_on=[1,2])
    XX = transform(X, scaler)
    @test mean(minimum(XX[[1,2],:], 2)) ≈ -2 
    @test mean(maximum(XX[[1,2],:], 2)) ≈ 2 

    scaler = fit(FixedRangeScaler, X, -2, 2, obsdim=2, operate_on=[1,2])
    XX = deepcopy(X)
    transform!(XX, scaler)
    @test mean(minimum(XX[[1,2],:], 2)) ≈ -2 
    @test mean(maximum(XX[[1,2],:], 2)) ≈ 2 
end

@testset "DataFrame" begin
    scaler = fit(FixedRangeScaler, D)
    DD = transform(D, scaler)
    @test minimum(DD[:A]) == 0 
    @test maximum(DD[:A]) == 1 

    scaler = fit(FixedRangeScaler, D , -1, 1)
    DD = transform(D, scaler)
    @test minimum(DD[:A]) == -1
    @test maximum(DD[:A]) == 1
    @test minimum(DD[:B]) == -1 
    @test maximum(DD[:B]) == 1 

    scaler = fit(FixedRangeScaler, D, -1, 1, operate_on=[:A])
    DD = transform(D, scaler)
    @test minimum(DD[:A]) == -1
    @test maximum(DD[:A]) == 1
    @test minimum(DD[:B]) == minimum(D[:B]) 
    @test maximum(DD[:B]) == maximum(D[:B])

    scaler = fit(FixedRangeScaler, D, -1, 1)
    DD = transform(D_NA, scaler)
    @test isna(DD[1,:A])
    @test DD[end,:A] == D_NA[end,:A]
    @test minimum(DD[:B]) == -1 
    @test maximum(DD[:B]) == 1 

    scaler = fit(FixedRangeScaler, D, -1, 1, operate_on=[:A, :B])
    DD = transform(D_NA, scaler)
    @test isna(DD[1,:A])
    @test DD[end,:A] == D_NA[end,:A]
    @test minimum(DD[:B]) == -1 
    @test maximum(DD[:B]) == 1 

    scaler = fit(FixedRangeScaler, D, -1, 1, operate_on=[:A, :B, :C])
    DD = transform(D_NA, scaler)
    @test isna(DD[1,:A])
    @test DD[end,:A] == D_NA[end,:A]
    @test minimum(DD[:B]) == -1 
    @test maximum(DD[:B]) == 1 

    DD = deepcopy(D)
    scaler = fit(FixedRangeScaler, D, -1, 1, operate_on=[:A, :B])
    transform!(DD, scaler)
    @test minimum(DD[:A]) == -1 
    @test maximum(DD[:A]) == 1 
    @test minimum(DD[:B]) == -1 
    @test maximum(DD[:B]) == 1 
end
