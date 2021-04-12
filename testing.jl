include("slow_sequential.jl")
# include("acquisition_exploration.jl")
include("testing_utilities.jl")

cov_grid = get_cov_grid(1,5)
println(length(cov_grid))
# println(display(cov_grid[5]))
# println(display(get_values_shallow(cov_grid[100])))

function run_inference(dataset_name, animation_name, n_particles, sequential, cov_fn)
    anim_traj = Dict()

    # set seed
    Random.seed!(1)

    # do inference and plot visualization
    if (sequential)
        # @time state = particle_filter_sequential(xs_train, ys_train, n_particles, pf_callback, anim_traj, xs_test, ys_test, cov_fn)
        # make_animation_sequential(animation_name, anim_traj, n_particles, xs_train, ys_train, xs, ys)
        obs_choices = choicemap()
        for t=1:length(ys_train)
            obs_choices[(:y, t)] = ys_train[t]
        end
        cov_fn_map = node_to_choicemap(cov_fn, choicemap(), 1)
        # display(merge(obs_choices, cov_fn_map))
        # println(merge(obs_choices, cov_fn_map))
        trace, weight = generate(model, Tuple([xs_train]), merge(obs_choices, cov_fn_map))
        likelihood = project(trace, select([(:y, i) for i=1:length(ys_train)]...))
        # display(get_choices(trace))
        # print(cov_fn)
        # println("    likelihood ", likelihood)
    # else
    #     x_obs_traj = Float64[]
    #     y_obs_traj = Float64[]
        # @time state = particle_filter_acquisition(xs_train, ys_train, n_particles, pf_callback, anim_traj, x_obs_traj, y_obs_traj)
        # make_animation_acquisition(animation_name, anim_traj, n_particles, xs_train, ys_train, xs, ys, x_obs_traj, y_obs_traj)
    end
    return (cov_fn, likelihood)
end


# dataset_names = ["airline"]
# dataset_names = ["quadratic", "changepoint", "polynomial"]
dataset_name = "sinusoid"
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
    n_particles = 1
    sequential = true
    animation_name = "sequential_" * dataset_name
    ret_likelihood = run_inference(dataset_name, animation_name, n_particles, sequential, cov_fn)
    push!(results, ret_likelihood)

    # run acquisition prediction
    # n_particles = 1
    # sequential = false
    # animation_name = "acq_exp_" * dataset_name
    # # animation_name = "acquisition_" * dataset_name
    # run_inference(dataset_name, animation_name, n_particles, sequential)
end

# sort by likelihood
sort!(results, by = x -> x[2], rev = true);
for ret in results
    print(ret[1])
    println("    likelihood ", ret[2])
end
animation_name = "testing_" * dataset_name
make_animation_likelihood(animation_name, results, xs_train, ys_train)
