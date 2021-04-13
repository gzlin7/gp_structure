include("slow_sequential.jl")
# include("acquisition_exploration.jl")
include("testing_utilities.jl")

cov_grid = get_cov_grid(1,10)
println("COV GRID LENGTH: ", length(cov_grid))
dataset_names = ["cubic", "quadratic", "changepoint", "polynomial", "sinusoid", "airline"]

function test_dataset(dataset_names, cov_grid)
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

        for i=1:length(cov_grid)
            cov_fn = cov_grid[i]
            # # run sequential prediction
            sequential = true
            ret_likelihood = run_inference(dataset_name, true, cov_fn, xs_train, ys_train)
            push!(results, ret_likelihood)
        end

        # sort by likelihood
        sort!(results, by = x -> x[2], rev = true);
        for ret in results
            print(ret[1])
            println("    likelihood ", ret[2])
        end
        animation_name = "testing_" * dataset_name
        make_animation_likelihood(animation_name, results, xs_train, ys_train)
    end
end

test_dataset(dataset_names, cov_grid)
