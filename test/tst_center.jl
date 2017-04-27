e_x = collect(-2:0.5:10)
e_X = expand_poly(e_x, 5)
df = DataFrame(A=rand(10), B=collect(1:10), C=[string(x) for x in 1:10])
df_na = deepcopy(df)
df_na[1, :A] = NA

@testset "Array" begin
    # Center Vectors
    xa = copy(e_x)
    @test center!(xa) ≈ mean(e_x)
    @test abs(mean(xa)) <= 10e-10

    xa = copy(e_x)
    mu = mean(xa)
    center!(xa, mu, obsdim=1)
    @test abs(mean(xa)) <= 10e-10

    xa = copy(e_x)
    mu = vec(ones(xa))
    center!(xa, mu, obsdim=1)
    @test sum(e_x .- mean(xa)) ≈ length(mu)

    # Center Matrix w/o mu
    Xa = copy(e_X)
    center!(Xa)
    @test abs(sum(mean(Xa, 2))) <= 10e-10

    Xa = copy(e_X)
    center!(Xa, obsdim=1)
    @test abs(sum(mean(Xa, 1))) <= 10e-10

    Xa = copy(e_X)
    center!(Xa, ObsDim.First())
    @test abs(sum(mean(Xa, 1))) <= 10e-10

    Xa = copy(e_X)
    center!(Xa, obsdim=2)
    @test abs(sum(mean(Xa, 2))) <= 10e-10

    Xa = copy(e_X)
    center!(Xa, ObsDim.Last())
    @test abs(sum(mean(Xa, 2))) <= 10e-10


    # Center Matrix with mu as input
    Xa = copy(e_X)
    mu = vec(mean(Xa, 1))
    center!(Xa, mu, obsdim=1)
    @test abs(sum(mean(Xa, 1))) <= 10e-10

    Xa = copy(e_X)
    mu = vec(mean(Xa, 2))
    center!(Xa, mu, obsdim=2)
    @test abs(sum(mean(Xa, 2))) <= 10e-10

    Xa = copy(e_X)
    mu = vec(mean(Xa, 2))
    center!(Xa, mu, ObsDim.Last())
    @test abs(sum(mean(Xa, 2))) <= 10e-10
end

@testset "DataFrame" begin
    # Center DataFrame
    D = copy(df)
    mu_check = [mean(D[colname]) for colname in names(D)[1:2]]
    mu = center!(D)
    @test length(mu) == 2
    @test abs(sum(mu .- mu_check)) <= 10e-10

    D = copy(df)
    mu_check = [mean(D[colname]) for colname in names(D)[1:2]]
    mu = center!(D, [:A, :B])
    @test abs(sum(mu .- mu_check)) <= 10e-10

    D = copy(df)
    mu_check = [mean(D[colname]) for colname in names(D)[1:2]]
    mu = center!(D, [:A, :B], mu_check)
    @test abs(sum([mean(D[colname]) for colname in names(D)[1:2]])) <= 10e-10

    # skip columns that contain NA values
    D = copy(df_na)
    mu = center!(D, [:A, :B])
    @test isna(D[1, :A])
    @test all(D[2:end, :A] .== df_na[2:end, :A])
    @test abs(mean(D[:B])) < 10e-10

    D = copy(df_na)
    mu_check = [mean(D[colname]) for colname in names(D)[1:2]]
    mu = center!(D, [:A, :B], mu_check)
    @test isna(D[1, :A])
    @test all(D[2:end, :A] .== df_na[2:end, :A])
    @test abs(mean(D[:B])) < 10e-10
end
