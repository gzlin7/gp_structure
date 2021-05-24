include("sequential.jl")
include("acquisition_exploration_AL.jl")


functions = Dict("sinusoid"=> x -> 0.20sin(15.7x),
                 "linear"=> x -> 0.4x,
                 "quadratic" => x -> 4(x - 2) ^ 2 - 2,
                 "polynomial" => x -> -0.2 * (x - 0.3)^2 * (x - 3.3) * (x - 4.2) * (x - 1.3) * (x - 3.7) * (x + 0.4) * (x - 2.1),
                 "cubic" => x -> 20 * ((x - 0.2)*(x - 0.6)*(x - 1)) + 0.5,
                 "changepoint" => x -> x < 2.0 ? 4(x - 1) ^ 2 : 0.20sin(15.7x),
                 "airline" => get_airline_dataset,
                 # http://infinity77.net/global_optimization/test_functions_1d.html
                 "02" => x -> sin(x) + sin(10/3 * x),
                 "03" => x -> -sum([k * sin((k + 1) * x + k) for k=1:6]),
                 "04" => x -> -(16*x^2 - 24*x + 5) * exp(-x),
                 "05" => x -> -(1.4-3*x)*sin(18*x),
                 "06" => x -> -(x + sin(x)) * exp(-x^2),
                 "07" => x -> sin(x) + sin(10/3 * x) + log(x) - 0.84*x + 3,
                 "08" => x -> -sum([k * cos((k + 1) * x + k) for k=1:6])                 )

bounds_default = (0.0,0.4)
bounds =  Dict( "02" =>  (2.7,7.5),
                "03" => (-10, 10),
                "04" => (1.9, 3.9),
                "05" => (0, 1.2),
                "06" => (-10, 10),
                "07" => (2.7, 7.5),
                "08" => (-10, 10),
                "09" => (3.1, 20.4)
                 )

n_obs_default = 100
fn_to_obs = Dict("linear"=> 50,
                 "quadratic" => 50,
                 "03" => 500,
                 "06" => 500
                  )



function run_inference(dataset_name, animation_name, n_particles, sequential, f, n_obs_plotting, budget, random)
    # load the dataset
    if (dataset_name == "airline")
        (xs, ys) = f()
        xs_train = xs[1:100]
        ys_train = ys[1:100]
        xs_test = xs_train
        ys_test = ys_train

    else
        # Generate data
        data_bounds = haskey(bounds, dataset_name) ? bounds[dataset_name] : bounds_default
        xs_train = collect(LinRange(data_bounds[1], data_bounds[2], n_obs_plotting))
        sort!(xs_train)
        ys_train = deepcopy(xs_train)
        @. ys_train = f.(xs_train)

        xs_test = [uniform(data_bounds[1], data_bounds[2]) for t=1:n_obs_plotting]
        sort!(xs_test)
        ys_test = deepcopy(xs_test)
        @. ys_test = f.(xs_test)
    end

    anim_traj = Dict()

    # set seed
    Random.seed!(1)

    function pf_callback(state, xs_train, ys_train, anim_traj, t)
        # calculate E[MSE]
        n_particles = length(state.traces)
        e_mse = 0
        e_pred_ll = 0
        weights = get_norm_weights(state)
        if haskey(anim_traj, t) == false
            push!(anim_traj, t => [])
        end
        for i=1:n_particles
            trace = state.traces[i]
            covariance_fn = get_retval(trace)[1]
            noise = trace[:noise]
            push!(anim_traj[t], [covariance_fn, noise, weights[i]])
            mse =  compute_mse(covariance_fn, noise, xs_train, ys_train, xs_test, ys_test)
            pred_ll = predictive_ll(covariance_fn, noise, xs_train, ys_train, xs_test, ys_test)
            e_mse += mse * weights[i]
            e_pred_ll += pred_ll * weights[i]
        end
        println("E[mse]: $e_mse, E[predictive log likelihood]: $e_pred_ll")
    end

    # do inference and plot visualization
    if (sequential)
        @time state = particle_filter_sequential(xs_train, ys_train, n_particles, pf_callback, anim_traj)
        make_animation_sequential(animation_name, anim_traj, n_particles, xs_train, ys_train, xs_test, ys_test)
    else
        x_obs_traj = Float64[]
        y_obs_traj = Float64[]
        @time state = particle_filter_acquisition_AL(xs_train, ys_train, n_particles, pf_callback, anim_traj, x_obs_traj, y_obs_traj, budget, random)
        make_animation_acquisition(animation_name, anim_traj, n_particles, xs_train, ys_train, xs_test, ys_test, x_obs_traj, y_obs_traj)
    end
end

# dataset_names = ["changepoint", "polynomial", "sinusoid", "quadratic", "linear","airline", "quadratic"]
dataset_names = ["05", "02", "airline"]
# dataset_names = ["quadratic"]
n_particles_all = [100]


# n_particles = 50
# dataset_name = "quadratic"
# sequential = true
# animation_name = "sequential_" * dataset_name * "_" * string(n_particles)
#
# ret = run_inference(dataset_name, animation_name, n_particles, sequential, functions[dataset_name], n_observations)
# @unpack animation_name, anim_traj, n_particles, xs_train, ys_train, xs_test, ys_test, x_obs_traj, y_obs_traj = ret;
#
# make_animation_acquisition(animation_name, anim_traj, n_particles, xs_train, ys_train, xs_test, ys_test, x_obs_traj, y_obs_traj)


for i=1:length(dataset_names)
    dataset_name = dataset_names[i]
    n_obs_plotting = haskey(fn_to_obs, dataset_name) ? fn_to_obs[dataset_name] : n_obs_default
    budget = 13

    # # run sequential prediction
    for n_particles in n_particles_all
        # sequential = true
        # animation_name = "sequential_" * dataset_name * "_" * string(n_particles)

        # run acquisition prediction
        sequential = false
        # random = true

        animation_name_rand = dataset_name * "_rand_" * string(n_particles)
        animation_name_al = dataset_name * "_active_" * string(n_particles)

        run_inference(dataset_name, animation_name_rand, n_particles, sequential, functions[dataset_name], n_obs_plotting, budget, true)
        run_inference(dataset_name, animation_name_al, n_particles, sequential, functions[dataset_name], n_obs_plotting, budget, false)
    end
end
