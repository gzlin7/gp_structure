include("slow_sequential.jl")
# include("acquisition_exploration.jl")
include("testing_utilities.jl")
n_buckets = 20
cov_grid = get_cov_grid(1,n_buckets)
noise_max = 3
println("COV GRID LENGTH: ", length(cov_grid))
# dataset_names = ["cubic", "quadratic", "changepoint", "polynomial", "sinusoid", "airline"]
dataset_names = ["sinusoid"]

function test_dataset(dataset_names, cov_grid)
    noise = 0.01

    for dataset_name in dataset_names
        if (dataset_name == "airline")
            (xs, ys) = get_airline_dataset()
        else
            (xs, ys) = get_dataset(dataset_name)
        end
        xs_train = xs[1:100]
        ys_train = ys[1:100]
        xs_test = xs[101:end]
        ys_test = ys[101:end]

        results = []
        sum_exp = 0

        for i=1:length(cov_grid)
            if mod(i,1000) == 0
                println(i)
            end
            cov_fn = cov_grid[i]
            # for noise in LinRange(1/n_buckets,noise_max,n_buckets)
            # # run sequential prediction
            sequential = true
            _, likelihood, noise = run_inference(dataset_name, true, cov_fn, xs_train, ys_train, noise)
            sum_exp += exp(likelihood)
            push!(results, (cov_fn, exp(likelihood), noise))
            # end
        end

        # sort by likelihood
        sort!(results, by = x -> x[2], rev = true);
        for ret in results
            # print(ret[1])
            # println("    likelihood ", ret[2])
        end
        animation_name = "testing_" * dataset_name
        make_animation_likelihood(animation_name, results, xs_train, ys_train, sum_exp)
    end
end

test_dataset(dataset_names, cov_grid)
