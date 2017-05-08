e_x = collect(-2:0.5:10)
e_X = expand_poly(e_x, 5)
df = DataFrame(A=rand(10), B=collect(1:10), C=[string(x) for x in 1:10])
df_na = deepcopy(df)
df_na[1, :A] = NA

@testset "Array" begin
    # Rescale Vector
    xa = copy(e_x)
    mu, sigma = standardize!(xa)
    @test mu ≈ mean(e_x)
    @test sigma ≈ std(e_x)
    @test abs(mean(xa)) <= 10e-10
    @test std(xa) ≈ 1

    xa = copy(e_x)
    mu, sigma = standardize!(xa, mu, sigma)
    @test abs(mean(xa)) <= 10e-10
    @test std(xa) ≈ 1

    xa = copy(e_x)
    mu, sigma = standardize!(xa, mu, sigma, obsdim=1)
    @test abs(mean(xa)) <= 10e-10
    @test std(xa) ≈ 1

    xa = copy(e_x)
    mu = copy(e_x) .- 1
    sigma = ones(e_x)
    mu, sigma = standardize!(xa, mu, sigma, obsdim=1)
    @test mean(xa) ≈ 1

    Xa = copy(e_X)
    standardize!(Xa)
    @test abs(sum(mean(Xa, 2))) <= 10e-10
    @test std(Xa, 2) ≈ [1, 1, 1, 1, 1]

    Xa = copy(e_X)
    standardize!(Xa, obsdim=2)
    @test abs(sum(mean(Xa, 2))) <= 10e-10
    @test std(Xa, 2) ≈ [1, 1, 1, 1, 1]

    Xa = copy(e_X)
    standardize!(Xa, obsdim=1)
    @test abs(sum(mean(Xa, 1))) <= 10e-10

    Xa = copy(e_X)
    mu = vec(mean(Xa, 1))
    sigma = vec(std(Xa, 1))
    standardize!(Xa, mu, sigma, obsdim=1)
    @test abs(sum(mean(Xa, 1))) <= 10e-10

    Xa = copy(e_X)
    mu = vec(mean(Xa, 2))
    sigma = vec(std(Xa, 2))
    standardize!(Xa, mu, sigma, obsdim=2)
    @test abs(sum(mean(Xa, 2))) <= 10e-10
end

#= @testset "DataFrame" begin =#
#=     D = copy(df) =#
#=     mu, sigma = standardize!(D) =#
#=     @test abs(sum([mean(D[colname]) for colname in names(D)[1:2]])) <= 10e-10 =#
#=     @test mean([std(D[colname]) for colname in names(D)[1:2]]) - 1 <= 10e-10 =#

#=     D = copy(df) =#
#=     mu, sigma = standardize!(D, [:A, :B]) =#
#=     @test abs(sum([mean(D[colname]) for colname in names(D)[1:2]])) <= 10e-10 =#
#=     @test mean([std(D[colname]) for colname in names(D)[1:2]]) - 1 <= 10e-10 =#

#=     D = copy(df) =#
#=     mu_check = [mean(D[colname]) for colname in names(D)[1:2]] =#
#=     sigma_check = [std(D[colname]) for colname in names(D)[1:2]] =#
#=     mu, sigma = standardize!(D, [:A, :B], mu_check, sigma_check) =#
#=     @test abs(sum([mean(D[colname]) for colname in names(D)[1:2]])) <= 10e-10 =#
#=     @test mean([std(D[colname]) for colname in names(D)[1:2]]) - 1 <= 10e-10 =#

#=     # skip columns that contain NA values =#
#=     D = copy(df_na) =#
#=     mu, sigma = standardize!(D, [:A, :B]) =#
#=     @test isna(D[1, :A]) =#
#=     @test all(D[2:end, :A] .== df_na[2:end, :A]) =#
#=     @test abs(mean(D[:B])) < 10e-10 =#
#=     @test abs(std(D[:B])) - 1 < 10e-10 =#

#=     D = copy(df_na) =#
#=     mu_check = [mean(D[colname]) for colname in names(D)[1:2]] =#
#=     sigma_check = [std(D[colname]) for colname in names(D)[1:2]] =#
#=     mu, sigma = standardize!(D, [:A, :B], mu_check, sigma_check) =#
#=     #1= @test isna(D[1, :A]) =1# =#
#=     #1= @test all(D[2:end, :A] .== df_na[2:end, :A]) =1# =#
#=     #1= @test abs(mean(D[:B])) < 10e-10 =1# =#
#=     #1= @test (abs(std(D[:B])) - 1) < 10e-10 =1# =#
#= end =#
