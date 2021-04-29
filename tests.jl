include("testing_utilities.jl")
using Test, Logging

functions = Dict("sinusoid"=> x -> 0.20sin(15.7x),
                 "linear"=> x -> 0.4x,
                 "quadratic" => x -> 4(x - 1) ^ 2)

noise_max = 3

@testset "Sequential GP (2 Layers)" begin
    n_buckets = 5
    cov_grid = get_cov_grid(2,n_buckets)
    println("COV GRID LENGTH: ", length(cov_grid))

    # scale 0.2 period 0.4
    @test Times(Periodic(1.0, 0.8), Constant(0.2)) in test_dataset("sinusoid", cov_grid, functions["sinusoid"], 100; animate=false)
    @test count(x -> x.left == Linear(0.4) || x.right == Linear(0.4), test_dataset("linear", cov_grid, functions["linear"], 100; animate=false)) > 0
    # @test Times(Periodic(1.0, 0.8), Constant(0.2)) in test_dataset("quadratic", cov_grid, functions["quadratic"], 100; animate=true, sequential=true)
end
