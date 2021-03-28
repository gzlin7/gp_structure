include("sequential.jl")
include("acquisition.jl")

function run_inference(dataset_name, animation_name, n_particles)
    # load the dataset
    (xs, ys) = get_dataset(dataset_name)
    # (xs, ys) = get_airline_dataset()
    xs_train = xs[1:100]
    ys_train = ys[1:100]
    xs_test = xs[101:end]
    ys_test = ys[101:end]

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
        make_animation_sequential(animation_name, anim_traj, xs_train, ys_train, xs, ys)
    else
        x_obs_traj = Float64[]
        y_obs_traj = Float64[]
        @time state = particle_filter_acquisition(xs_train, ys_train, n_particles, pf_callback, anim_traj, x_obs_traj, y_obs_traj)
        make_animation_acquisition(animation_name, anim_traj, xs_train, ys_train, xs, ys, x_obs_traj, y_obs_traj)
    end
end

dataset_names = ["quadratic", "cubic"]

for i=1:length(dataset_names)
    dataset_name = dataset_names[i]

    # run sequential prediction
    n_particles = 100
    sequential = true
    animation_name = "sequential_" * dataset_name
    run_inference(dataset_name, animation_name, n_particles)

    # # run acquisition prediction
    # n_particles = 50
    # sequential = false
    # animation_name = "acquisition_" * dataset_name
    # run_inference(dataset_name, animation_name, n_particles)
end
