X = collect(Float64, reshape(1:40, 10, 4))
x = rand(10) * 10
w = collect(Float64, 1:10)

D = DataFrame(A=rand(10), B=collect(1:10), C=[hex(x) for x in 11:20])
D_NA = deepcopy(D)
D_NA[1, :A] = NA
@testset "Array" begin
    scaler = FixedRangeScaler(X)
    XX = transform(X, scaler)
    @test mean(XX[:,end]) ≈ 1
    @test mean(XX[:,1]) ≈ 0
    @test maximum(XX) == 1
    @test minimum(XX) == 0

    XX, scaler = fit_transform(FixedRangeScaler, X)
    @test mean(XX[:,end]) ≈ 1
    @test mean(XX[:,1]) ≈ 0
    @test maximum(XX) == 1
    @test minimum(XX) == 0

    XX = deepcopy(X)
    scaler = fit_transform!(FixedRangeScaler, XX)
    @test mean(XX[:,end]) ≈ 1
    @test mean(XX[:,1]) ≈ 0
    @test maximum(XX) == 1
    @test minimum(XX) == 0

    scaler = FixedRangeScaler(X, -2, 2)
    XX = transform(X, scaler)
    @test mean(XX[:,end]) ≈ 2
    @test mean(XX[:,1]) ≈ -2 
    @test maximum(XX) == 2 
    @test minimum(XX) == -2 

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

    XX, scaler = fit_transform(FixedRangeScaler, X, -2, 2)
    @test mean(XX[:,end]) ≈ 2
    @test mean(XX[:,1]) ≈ -2
    @test maximum(XX) == 2
    @test minimum(XX) == -2 

    XX = deepcopy(X)
    scaler = fit_transform!(FixedRangeScaler, XX, -2, 2)
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
    ww = deepcopy(w)
    transform!(ww, scaler)
    @test ww[1] == -2
    @test all(w[3:end] .== ww[3:end])

    scaler = fit(FixedRangeScaler, X, -2, 2, obsdim=2, operate_on=[1,2])
    XX = transform(X, scaler)
    @test mean(minimum(XX[[1,2],:], 2)) ≈ -2 
    @test mean(maximum(XX[[1,2],:], 2)) ≈ 2 

    Xi = round(Int, X)
    transform(X, scaler)
    @test mean(minimum(XX[[1,2],:], 2)) ≈ -2 
    @test mean(maximum(XX[[1,2],:], 2)) ≈ 2 

    scaler = fit(FixedRangeScaler, X, -2, 2, obsdim=2, operate_on=[1,2])
    XX = deepcopy(X)
    transform!(XX, scaler)
    @test mean(minimum(XX[[1,2],:], 2)) ≈ -2 
    @test mean(maximum(XX[[1,2],:], 2)) ≈ 2 

    XX = deepcopy(X)
    fixedrange!(XX)
    @test all(maximum(XX, 2) .== ones(size(XX, 1)))
    @test all(minimum(XX, 2) .== zeros(size(XX, 1)))

    XX = deepcopy(X)
    fixedrange!(XX, ObsDim.Last(), collect(1:size(X,1)))
    @test all(maximum(XX, 2) .== ones(size(XX, 1)))
    @test all(minimum(XX, 2) .== zeros(size(XX, 1)))

    XX = deepcopy(X)
    fixedrange!(XX, -2, 2)
    @test all(maximum(XX, 2) .== ones(size(XX, 1)) .* 2)
    @test all(minimum(XX, 2) .== ones(size(XX, 1)) .* -2)

    XX = deepcopy(X)
    fixedrange!(XX, 0, 1, ObsDim.First(), collect(1:size(X, 2)))
    @test all(maximum(XX, 1) .== ones(size(XX, 2)))
    @test all(minimum(XX, 1) .== zeros(size(XX, 2)))

    XX = deepcopy(X)
    fixedrange!(XX, 0, 1, ObsDim.Last(), collect(1:size(X,1)))
    @test all(maximum(XX, 2) .== ones(size(XX, 1)))
    @test all(minimum(XX, 2) .== zeros(size(XX, 1)))

    XX = deepcopy(X)
    fixedrange!(XX, 0, 1, vec(minimum(XX, 2)), vec(maximum(XX, 2)))
    @test all(maximum(XX, 2) .== ones(size(XX, 1)))
    @test all(minimum(XX, 2) .== zeros(size(XX, 1)))


    XX = deepcopy(X)
    fixedrange!(XX, 0, 1, vec(minimum(XX, 2)), vec(maximum(XX, 2)), ObsDim.Last(), collect(1:size(XX, 1)))
    @test all(maximum(XX, 2) .== ones(size(XX, 1)))
    @test all(minimum(XX, 2) .== zeros(size(XX, 1)))

    XX = deepcopy(X)
    fixedrange!(XX, 0, 1, vec(minimum(XX, 2)), vec(maximum(XX, 2)), ObsDim.Constant{2}(), collect(1:size(XX, 1)))
    @test all(maximum(XX, 2) .== ones(size(XX, 1)))
    @test all(minimum(XX, 2) .== zeros(size(XX, 1)))

    xx = deepcopy(x)
    fixedrange!(xx)
    @test minimum(xx) == 0
    @test maximum(xx) == 1

    xx = deepcopy(x)
    fixedrange!(xx, -1, 1)
    @test minimum(xx) == -1
    @test maximum(xx) == 1

    xx = deepcopy(x)
    xmin = minimum(x, 2) .- 1
    xmax = maximum(x, 2)

    xx = deepcopy(x)
    fixedrange!(xx, -1, 1, xmin, xmax, ObsDim.First(), collect(1:length(x)))
    @test all(minimum(xx, 2) .== ones(length(x)))
    @test all(maximum(xx, 2) .== ones(length(x)))

    xx = deepcopy(x)
    fixedrange!(xx, -1, 1, xmin, xmax, ObsDim.Last(), collect(1:length(x)))
    @test all(minimum(xx, 2) .== ones(length(x)))
    @test all(maximum(xx, 2) .== ones(length(x)))
end


@testset "DataFrame" begin
    scaler = FixedRangeScaler(D)
    DD = transform(D, scaler)
    @test minimum(DD[:A]) == 0 
    @test maximum(DD[:A]) == 1 
    @test minimum(DD[:B]) == 0 
    @test maximum(DD[:B]) == 1 
    
    scaler = fit(FixedRangeScaler, D)
    DD = transform(D, scaler)
    @test minimum(DD[:A]) == 0 
    @test maximum(DD[:A]) == 1 
    @test minimum(DD[:B]) == 0 
    @test maximum(DD[:B]) == 1 

    DD, scaler = fit_transform(FixedRangeScaler, D)
    @test minimum(DD[:A]) == 0 
    @test maximum(DD[:A]) == 1 
    @test minimum(DD[:B]) == 0 
    @test maximum(DD[:B]) == 1 

    DD = deepcopy(D)
    scaler = fit_transform!(FixedRangeScaler, DD)
    @test minimum(DD[:A]) == 0 
    @test maximum(DD[:A]) == 1 
    @test minimum(DD[:B]) == 0 
    @test maximum(DD[:B]) == 1 

    scaler = fit(FixedRangeScaler, D, -1, 1)
    DD = transform(D, scaler)
    @test minimum(DD[:A]) == -1
    @test maximum(DD[:A]) == 1
    @test minimum(DD[:B]) == -1 
    @test maximum(DD[:B]) == 1 

    DD, scaler = fit_transform(FixedRangeScaler, D, -1, 1)
    @test minimum(DD[:A]) == -1
    @test maximum(DD[:A]) == 1
    @test minimum(DD[:B]) == -1 
    @test maximum(DD[:B]) == 1 

    DD = deepcopy(D)
    scaler = fit_transform!(FixedRangeScaler, DD, -1, 1)
    @test minimum(DD[:A]) == -1
    @test maximum(DD[:A]) == 1
    @test minimum(DD[:B]) == -1 
    @test maximum(DD[:B]) == 1 

    scaler = FixedRangeScaler(D, -1, 1)
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


    DD = deepcopy(D)
    fixedrange!(DD)
    @test minimum(DD[:A]) == 0 
    @test maximum(DD[:A]) == 1 
    @test minimum(DD[:B]) == 0 
    @test maximum(DD[:B]) == 1 

    DD = deepcopy(D)
    fixedrange!(DD, -1, 1)
    @test minimum(DD[:A]) == -1 
    @test maximum(DD[:A]) == 1 
    @test minimum(DD[:B]) == -1 
    @test maximum(DD[:B]) == 1 

    DD = deepcopy(D)
    fixedrange!(DD, -1, 1, [:A, :B, :C])
    @test minimum(DD[:A]) == -1 
    @test maximum(DD[:A]) == 1 
    @test minimum(DD[:B]) == -1 
    @test maximum(DD[:B]) == 1 

    DD = deepcopy(D_NA)
    fixedrange!(DD, -1, 1, [:A, :B, :C])
    @test minimum(DD[:B]) == -1 
    @test maximum(DD[:B]) == 1 

    DD = deepcopy(D)
    fixedrange!(DD, -1, 1, [0,1], [1,10])
    @test minimum(DD[:B]) == -1 
    @test maximum(DD[:B]) == 1 
end
