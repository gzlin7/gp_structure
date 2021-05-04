include("sequential.jl")
include("acquisition_exploration.jl")


functions = Dict("sinusoid"=> x -> 0.20sin(15.7x),
                 "linear"=> x -> 0.4x,
                 "quadratic" => x -> 4(x - 2) ^ 2 - 2,
                 "polynomial" => x -> -0.2 * (x - 0.3)^2 * (x - 3.3) * (x - 4.2) * (x - 1.3) * (x - 3.7) * (x + 0.4) * (x - 2.1),
                 "cubic" => x -> 20 * ((x - 0.2)*(x - 0.6)*(x - 1)) + 0.5,
                 "changepoint" => x -> x < 2.0 ? 4(x - 1) ^ 2 : 0.20sin(15.7x),
                 "airline" => get_airline_dataset
                 )

 fn_to_obs = Dict("sinusoid"=> 100,
                  "linear"=> 50,
                  "quadratic" => 50,
                  "polynomial" => 100,
                  "cubic" => 100,
                  "changepoint" => 100,
                  "airline" => 100
                  )



function run_inference(dataset_name, animation_name, n_particles, sequential, f, n_obs)
    # load the dataset
    if (dataset_name == "airline")
        (xs, ys) = f()
        xs_train = xs[1:100]
        ys_train = ys[1:100]
        xs_test = xs_train
        ys_test = ys_train

    else
        # Generate data
        xs_train = collect(LinRange(0.0,4.0,n_obs))
        sort!(xs_train)
        ys_train = deepcopy(xs_train)
        @. ys_train = f.(xs_train)

        xs_test = [uniform(0.0,4.0) for t=1:n_obs]
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
        @time state = particle_filter_acquisition(xs_train, ys_train, n_particles, pf_callback, anim_traj, x_obs_traj, y_obs_traj)
        make_animation_acquisition(animation_name, anim_traj, n_particles, xs_train, ys_train, xs_test, ys_test, x_obs_traj, y_obs_traj)
    end
end

dataset_names = ["changepoint", "polynomial", "sinusoid", "quadratic", "linear","airline", "quadratic"]
# dataset_names = ["quadratic"]
n_particles_all = [50, 100]


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
    n_observations = fn_to_obs[dataset_name]

    # # run sequential prediction
    for n_particles in n_particles_all
        sequential = true
        animation_name = "sequential_" * dataset_name * "_" * string(n_particles)

        # run acquisition prediction
        # sequential = false
        # animation_name = "acquisition_" * dataset_name * "_" * string(n_particles)
        # animation_name = "acquisition_" * dataset_name

        run_inference(dataset_name, animation_name, n_particles, sequential, functions[dataset_name], n_observations)
    end
end
