X = collect(Float64, reshape(1:40, 10, 4))
x = rand(10) * 10

D = DataFrame(A=rand(10), B=collect(1:10), C=[hex(x) for x in 11:20])
D_NA = deepcopy(D)
D_NA[1, :A] = NA

@testset "Array" begin
    # Rescale Vector
    xx = deepcopy(x)
    mu, sigma = standardize!(xx)
    @test mu ≈ mean(x)
    @test sigma ≈ std(x)
    @test abs(mean(xx)) <= 10e-10
    @test std(xx) ≈ 1

    xx = deepcopy(x)
    mu, sigma = standardize!(xx, mu, sigma)
    @test abs(mean(xx)) <= 10e-10
    @test std(xx) ≈ 1

    xx = deepcopy(x)
    mu, sigma = standardize!(xx, mu, sigma, obsdim=1)
    @test abs(mean(xx)) <= 10e-10
    @test std(xx) ≈ 1

    xx = deepcopy(x)
    mu = deepcopy(x) .- 1
    sigma = ones(x)
    mu, sigma = standardize!(xx, mu, sigma, obsdim=1)
    @test mean(xx) ≈ 1

    # Rescale Matrix
    XX = deepcopy(X)
    standardize!(XX)
    @test abs(sum(mean(XX, 2))) <= 10e-10
    @test std(XX, 2) ≈ ones(size(X, 1)) 

    XX = deepcopy(X)
    standardize!(XX, obsdim=2)
    @test abs(sum(mean(XX, 2))) <= 10e-10
    @test std(XX, 2) ≈ ones(size(X, 1)) 

    XX = deepcopy(X)
    standardize!(XX, obsdim=1)
    @test abs(sum(mean(XX, 1))) <= 10e-10
    @test vec(std(XX, 1)) ≈ ones(size(X, 2)) 

    XX = deepcopy(X)
    mu = vec(mean(XX, 1))
    sigma = vec(std(XX, 1))
    standardize!(XX, mu, sigma, obsdim=1)
    @test abs(sum(mean(XX, 1))) <= 10e-10

    XX = deepcopy(X)
    mu = vec(mean(XX, 2))
    sigma = vec(std(XX, 2))
    standardize!(XX, mu, sigma, obsdim=2)
    @test abs(sum(mean(XX, 2))) <= 10e-10

    XX = deepcopy(X)
    flt = [1,2]
    standardize!(XX, obsdim=1, operate_on=flt)
    @test abs(sum(mean(XX[:,flt], 1))) <= 10e-10
    @test vec(std(XX[:,flt], 1)) ≈ ones(2) 
    @test all(X[:,[3,4]] .== XX[:,[3,4]])

    XX = deepcopy(X)
    flt = [2,8]
    mu = vec(mean(XX, 2))
    sigma = vec(std(XX, 2))
    standardize!(XX, mu[flt], sigma[flt], obsdim=2, operate_on=flt)
    @test abs(sum(mean(XX[flt,:], 2))) <= 10e-10

    scaler = StandardScaler(X)
    XX = transform(X, scaler)
    @test abs(sum(mean(XX, 2))) <= 10e-10
    @test std(XX, 2) ≈ ones(size(X, 1)) 

    scaler = fit(StandardScaler, X)
    XX = transform(X, scaler)
    @test abs(sum(mean(XX, 2))) <= 10e-10
    @test std(XX, 2) ≈ ones(size(X, 1)) 

    Xi = round(Int, X)
    XX = transform(Xi, scaler)
    @test abs(sum(mean(XX, 2))) <= 10e-10
    @test std(XX, 2) ≈ ones(size(X, 1)) 

    XX, scaler = fit_transform(StandardScaler, X)
    @test abs(sum(mean(XX, 2))) <= 10e-10
    @test std(XX, 2) ≈ ones(size(X, 1)) 

    XX = deepcopy(X)
    scaler = fit_transform!(StandardScaler, XX)
    @test abs(sum(mean(XX, 2))) <= 10e-10
    @test std(XX, 2) ≈ ones(size(X, 1)) 

    scaler = fit(StandardScaler, X, obsdim=2)
    XX = transform(X, scaler)
    @test abs(sum(mean(XX, 2))) <= 10e-10
    @test std(XX, 2) ≈ ones(size(X, 1)) 

    scaler = fit(StandardScaler, X, obsdim=1)
    XX = transform(X, scaler)
    @test abs(sum(mean(XX, 1))) <= 10e-10
    @test vec(std(XX, 1)) ≈ ones(size(X, 2)) 

    flt = [1,4]
    scaler = fit(StandardScaler, X, obsdim=1, operate_on=flt)
    XX = transform(X, scaler)
    xx = transform(vec(X[1,:]), scaler)
    @test abs(sum(mean(XX[:,flt], 1))) <= 10e-10
    @test vec(std(XX[:,flt], 1)) ≈ ones(size(X[:,flt], 2)) 
    @test all(xx .== XX[1,:])

    XX = deepcopy(X)
    xx = vec(X[1,:])
    flt = [1,4]
    scaler = fit(StandardScaler, X, obsdim=1, operate_on=flt)
    transform!(XX, scaler)
    transform!(xx, scaler)
    @test abs(sum(mean(XX[:,flt], 1))) <= 10e-10
    @test vec(std(XX[:,flt], 1)) ≈ ones(size(X[:,flt], 2)) 
    @test all(xx .== XX[1,:])
end

@testset "DataFrame" begin
    DD = deepcopy(D)
    mu, sigma = standardize!(DD)
    @test abs(sum([mean(DD[colname]) for colname in names(DD)[1:2]])) <= 10e-10
    @test mean([std(DD[colname]) for colname in names(DD)[1:2]]) - 1 <= 10e-10

    DD = deepcopy(D)
    mu, sigma = standardize!(DD, operate_on=[:A,:B,:C])
    @test abs(sum([mean(DD[colname]) for colname in names(DD)[1:2]])) <= 10e-10
    @test mean([std(DD[colname]) for colname in names(DD)[1:2]]) - 1 <= 10e-10

    DD = deepcopy(D_NA)
    m, s = standardize!(DD, [:A,:B,:C])
    @test all(DD[2:end, :A] .== D_NA[2:end, :A])
    @test all(DD[:C] .== D_NA[:C])
    @test mean(DD[:B]) ≈ 0.0

    DD = deepcopy(D_NA)
    m = 0.0
    s = 1.0
    m, s = standardize!(DD, m, s, :A)
    @test all(DD[2:end, :A] .== D_NA[2:end, :A])
    @test all(DD[:C] .== D_NA[:C])

    DD = deepcopy(D)
    mu, sigma = standardize!(DD, mu, sigma, operate_on=[:A,:B])
    @test abs(sum([mean(DD[colname]) for colname in names(DD)[1:2]])) <= 10e-10
    @test mean([std(DD[colname]) for colname in names(DD)[1:2]]) - 1 <= 10e-10

    # skip columns that contain NA values
    DD = deepcopy(D_NA)
    mu, sigma = standardize!(DD)
    @test isna(DD[1, :A])
    @test all(DD[2:end, :A] .== D_NA[2:end, :A])
    @test abs(mean(DD[:B])) < 10e-10
    @test abs(std(DD[:B])) - 1 < 10e-10

    scaler = StandardScaler(D)
    DD = transform(D, scaler)
    @test mean(DD[:A]) <= 10e-10 
    @test std(DD[:A]) - 1  <= 10e-10 
    @test mean(DD[:B]) <= 10e-10 
    @test std(DD[:B]) - 1  <= 10e-10 
    @test all(DD[:C] .== D[:C])

    scaler = fit(StandardScaler, D)
    DD = transform(D, scaler)
    @test mean(DD[:A]) <= 10e-10 
    @test std(DD[:A]) - 1  <= 10e-10 
    @test mean(DD[:B]) <= 10e-10 
    @test std(DD[:B]) - 1  <= 10e-10 
    @test all(DD[:C] .== D[:C])

    DD, scaler = fit_transform(StandardScaler, D)
    @test mean(DD[:A]) <= 10e-10 
    @test std(DD[:A]) - 1  <= 10e-10 
    @test mean(DD[:B]) <= 10e-10 
    @test std(DD[:B]) - 1  <= 10e-10 
    @test all(DD[:C] .== D[:C])

    DD = deepcopy(D)
    scaler = fit_transform!(StandardScaler, DD)
    @test mean(DD[:A]) <= 10e-10 
    @test std(DD[:A]) - 1  <= 10e-10 
    @test mean(DD[:B]) <= 10e-10 
    @test std(DD[:B]) - 1  <= 10e-10 
    @test all(DD[:C] .== D[:C])

    colnames = [:A, :B]
    offset = Float64[mean(D[colname]) for colname in colnames]
    scale = Float64[std(D[colname]) for colname in colnames]
    scaler = StandardScaler(D, offset, scale)
    @test mean(DD[:A]) <= 10e-10 
    @test std(DD[:A]) - 1  <= 10e-10 
    @test mean(DD[:B]) <= 10e-10 
    @test std(DD[:B]) - 1  <= 10e-10 
    @test all(DD[:C] .== D[:C])

    scaler = fit(StandardScaler, D, operate_on=[:A, :C])
    DD = transform(D, scaler)
    @test mean(DD[:A]) <= 10e-10 
    @test std(DD[:A]) - 1  <= 10e-10 
    @test all(DD[:B] .== D[:B])
    @test all(DD[:C] .== D[:C])
    @test mean(D[:A]) != mean(DD[:A]) 

    DD = deepcopy(D)
    scaler = fit(StandardScaler, DD, operate_on=[:A, :C])
    transform!(DD, scaler)
    @test mean(DD[:A]) <= 10e-10 
    @test std(DD[:A]) - 1  <= 10e-10 
    @test all(DD[:B] .== D[:B])
    @test all(DD[:C] .== D[:C])
    @test mean(D[:A]) != mean(DD[:A]) 
end
