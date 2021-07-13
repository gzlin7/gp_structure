include("../sequential.jl")
include("../acquisition_exploration_AL.jl")
include("functions.jl")
include("analytical_infogain.jl")
# include("../acquisition_exploration_testing.jl")

function load_data(dataset_name, n_obs_plotting)
    f = functions[dataset_name]
    if (dataset_name == "airline")
        (xs, ys) = f()
        xs_train = xs[1:100]
        ys_train = ys[1:100]
    else
        # Generate data
        data_bounds = haskey(bounds, dataset_name) ? bounds[dataset_name] : bounds_default
        xs_train = collect(LinRange(data_bounds[1], data_bounds[2], n_obs_plotting))

        if (dataset_name == "quadratic")
            xs_train = [[0.0, 2.0] ; xs_train]
        end

        sort!(xs_train)
        ys_train = deepcopy(xs_train)
        @. ys_train = f.(xs_train)
    end

    return (xs_train, ys_train)
end


function run_inference(data, n_particles, budget, random)
    # set seed
    # Random.seed!(1)

    xs_train, ys_train = data

    # define callback with data
    function pf_callback(state, xs_obs, ys_obs, anim_traj, t, xs_info_plot, info_plot)
        # calculate E[MSE]
        n_particles = length(state.traces)
        e_mse = 0
        e_pred_ll = 0
        weights = get_norm_weights(state)
        if haskey(anim_traj, t) == false
            push!(anim_traj, t => [])
        end
        if haskey(anim_traj, "e_mse") == false
            push!(anim_traj, "e_mse" => [])
            push!(anim_traj, "e_pred_ll" => [])
        end
        for i=1:n_particles
            trace = state.traces[i]
            covariance_fn = get_retval(trace)[1]
            # println(covariance_fn)
            noise = trace[:noise]
            mse = compute_mse(covariance_fn, noise, xs_obs, ys_obs, xs_train, ys_train)
            pred_ll = predictive_ll(covariance_fn, noise, xs_obs, ys_obs, xs_train, ys_train)
            e_mse += mse * weights[i]
            e_pred_ll += pred_ll * weights[i]
            push!(anim_traj[t], [covariance_fn, noise, weights[i], mse, pred_ll])
        end
        # store information gain plotting info for each timestep
        push!(anim_traj[t], [xs_info_plot, info_plot])
        # store E[MSE] and E[Predictive Likelihood]
        # println("E[mse]: $e_mse, E[predictive log likelihood]: $e_pred_ll")
        push!(anim_traj["e_mse"], e_mse)
        push!(anim_traj["e_pred_ll"], e_pred_ll)
    end

    anim_traj = Dict()
    x_obs_traj = Float64[]
    y_obs_traj = Float64[]
    @time state = particle_filter_acquisition_AL(xs_train, ys_train, n_particles, pf_callback, anim_traj, x_obs_traj, y_obs_traj, budget, random)
    return (anim_traj, n_particles, xs_train, ys_train, x_obs_traj, y_obs_traj)
end


function visualize_inference(animation_name, objective, info_metric, inference_ret)
    anim_traj, n_particles, xs_train, ys_train, x_obs_traj, y_obs_traj = inference_ret

    make_animation_acquisition(animation_name, objective, info_metric, anim_traj, n_particles, xs_train, ys_train, x_obs_traj, y_obs_traj)
    plot_name = animation_name * "_acc"
    make_accuracy_plot(plot_name, anim_traj)
end

# plot 1 function
# dataset_name, animation_name, n_particles, n_obs_plotting, budget = "airline", "airline_new", 100, 100, 15
# load data
# data = load_data(dataset_name, n_obs_plotting)

# cov_fn = SquaredExponential(0.1)
# cov_fn = Periodic(0.25, 0.5)
# pred_xs = sort!(collect(LinRange(-1, 1, 200)))
# # pred_xs = sort!(collect(LinRange(-10, 10, 200)))
# noise = 0.0001
# train_x, train_y = get_dataset_gp(cov_fn, noise, pred_xs)
# test_x, test_y = get_dataset_gp(cov_fn, noise, pred_xs)
# data = [train_x, train_y, test_x, test_y]

# run inference
# inference_ret = run_inference(data, n_particles, budget, "gp")
# # visualize
# visualize_inference(animation_name, inference_ret)
