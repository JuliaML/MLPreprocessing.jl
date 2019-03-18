X = collect(Float64, reshape(1:40, 10, 4))
x = rand(10) * 10

D = DataFrame(A=rand(10), B=1:10, C=string.(11:20; base=16))
D_NA = allowmissing!(copy(D))
D_NA[1, :A] = missing

@testset "Array" begin
    XX = deepcopy(X)
    mu = center!(XX, obsdim=1)
    @test sum(abs.(mean(XX; dims=1))) == 0
    @test all(std(XX; dims=1) .== std(X; dims=1))
    @test all(mu .== vec(mean(X; dims=1)))

    XX = deepcopy(X)
    mu = center!(XX, ObsDim.First())
    @test sum(abs.(mean(XX; dims=1))) == 0
    @test all(std(XX; dims=1) .== std(X; dims=1))
    @test all(mu .== vec(mean(X; dims=1)))

    XX = deepcopy(X)
    mu = center!(XX, ObsDim.Last())
    @test sum(abs.(mean(XX; dims=2))) == 0
    @test all(std(XX; dims=2) .== std(X; dims=2))
    @test all(mu .== vec(mean(X; dims=2)))

    XX = deepcopy(X)
    mu = center!(XX)
    @test sum(abs.(mean(XX; dims=2))) == 0
    @test all(std(XX; dims=2) .== std(X; dims=2))
    @test all(mu .== vec(mean(X; dims=2)))

    XX = deepcopy(X)
    mu = vec(mean(X; dims=1))
    center!(XX, mu, obsdim=1)
    @test sum(abs.(mean(XX; dims=1))) == 0
    @test all(std(XX; dims=1) .== std(X; dims=1))

    XX = deepcopy(X)
    mu = vec(mean(X; dims=1))
    center!(XX, mu, ObsDim.First())
    @test sum(abs.(mean(XX; dims=1))) == 0
    @test all(std(XX; dims=1) .== std(X; dims=1))

    XX = deepcopy(X)
    mu = vec(mean(XX; dims=2))
    center!(XX, mu, obsdim=2)
    @test sum(abs.(mean(XX; dims=2))) == 0
    @test all(std(XX; dims=2) .== std(X; dims=2))

    XX = deepcopy(X)
    mu = vec(mean(XX; dims=2))
    center!(XX, mu, ObsDim.Last())
    @test sum(abs.(mean(XX; dims=2))) == 0
    @test all(std(XX; dims=2) .== std(X; dims=2))

    XX = deepcopy(X)
    mu = vec(mean(X[:,[1,3]]; dims=1))
    center!(XX, mu, obsdim=1, operate_on=[1, 3])
    @test sum(abs.(mean(XX[:,[1,3]]; dims=1))) == 0
    @test all(XX[:,2] .== X[:,2])
    @test all(std(XX; dims=1) .== std(X; dims=1))

    XX = deepcopy(X)
    mu = vec(mean(X[[1,3],:]; dims=2))
    center!(XX, mu, obsdim=2, operate_on=[1, 3])
    @test sum(abs.(mean(XX[[1,3],:]; dims=2))) == 0
    @test all(XX[2,:] .== X[2,:])
    @test all(std(XX; dims=2) .== std(X; dims=2))
    println()

    xx = deepcopy(x)
    center!(xx)
    @test mean(xx) <= 10e-10

    xx = deepcopy(x)
    mu = mean(xx)
    center!(xx, mu)
    @test mean(xx) <= 10e-10

    xx = deepcopy(x)
    mu = ones(size(xx))
    center!(xx, mu)
    @test mean(xx) - mean(x) ≈ -1

    xx = deepcopy(x)
    mu = ones(size(xx))
    center!(xx, mu)
    @test mean(xx) - mean(x) ≈ -1
end

@testset "DataFrame" begin
    # Center DataFrame
    DD = deepcopy(D)
    center!(DD)
    @test abs.(mean(DD[:A])) <= 10e-10
    @test abs.(mean(DD[:B])) <= 10e-10
    @test all(DD[:C] .== D[:C])

    DD = deepcopy(D)
    center!(DD, operate_on=[:B])
    @test all(DD[:A] .== D[:A])
    @test abs.(mean(DD[:B])) <= 10e-10
    @test all(DD[:C] .== D[:C])

    DD = deepcopy(D)
    mu = center!(DD, operate_on=[:A, :B])
    @test abs.(mean(DD[:A])) <= 10e-10
    @test abs.(mean(DD[:B])) <= 10e-10
    @test all(DD[:C] .== D[:C])
    @test all(mu .== [mean(D[:A]), mean(D[:B])])

    DD = deepcopy(D_NA)
    mu = center!(DD, [:A, :B, :C])
    @test abs.(mean(DD[:B])) <= 10e-10
    @test all(DD[2:end, :A] .== D[2:end, :A])
    @test all(DD[:C] .== D[:C])

    DD = deepcopy(D_NA)
    mu = [0.0, mean(DD[:B]), 0.0]
    mu = center!(DD, mu, [:A, :B, :C])
    @test abs.(mean(DD[:B])) <= 10e-10
    @test all(DD[2:end, :A] .== D[2:end, :A])
    @test all(DD[:C] .== D[:C])

    DD = deepcopy(D_NA)
    mu = 0.0
    mu = center!(DD, mu, :A)
    @test all(DD[2:end, :A] .== D[2:end, :A])
    @test all(DD[:B] .== D[:B])
    @test all(DD[:C] .== D[:C])

    DD = deepcopy(D)
    mu =  [mean(D[:A]), mean(D[:B])]
    @test all(center!(DD, mu, operate_on=[:A, :B]) .== mu)
    @test abs.(mean(DD[:A])) <= 10e-10
    @test abs.(mean(DD[:B])) <= 10e-10
    @test all(DD[:C] .== D[:C])
    
    DD = deepcopy(D_NA)
    center!(DD)
    @test all(DD[2:end, :A] .== D[2:end, :A])
    @test abs.(mean(DD[:B])) <= 10e-10
    @test all(DD[:C] .== D[:C])
    @test ismissing(DD[1, :A])
end
