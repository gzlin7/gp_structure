# include("sequential.jl")
# include("acquisition_exploration_AL.jl")
# include("utils/functions.jl")
# # include("acquisition_exploration_testing.jl")
#
#
# function run_inference(dataset_name, animation_name, n_particles, f, n_obs_plotting, budget, random)
#     # load the dataset
#     if (dataset_name == "airline")
#         (xs, ys) = f()
#         xs_train = xs[1:100]
#         ys_train = ys[1:100]
#         xs_test = xs_train
#         ys_test = ys_train
#
#     else
#         # Generate data
#         data_bounds = haskey(bounds, dataset_name) ? bounds[dataset_name] : bounds_default
#         xs_train = collect(LinRange(data_bounds[1], data_bounds[2], n_obs_plotting))
#         sort!(xs_train)
#         ys_train = deepcopy(xs_train)
#         @. ys_train = f.(xs_train)
#
#         xs_test = [uniform(data_bounds[1], data_bounds[2]) for t=1:n_obs_plotting]
#         sort!(xs_test)
#         ys_test = deepcopy(xs_test)
#         @. ys_test = f.(xs_test)
#     end
#
#     anim_traj = Dict()
#
#     # set seed
#     Random.seed!(1)
#
#     function pf_callback(state, xs_obs, ys_obs, anim_traj, t, xs_info_plot, info_plot)
#         # calculate E[MSE]
#         n_particles = length(state.traces)
#         e_mse = 0
#         e_pred_ll = 0
#         weights = get_norm_weights(state)
#         if haskey(anim_traj, t) == false
#             push!(anim_traj, t => [])
#         end
#         if haskey(anim_traj, "e_mse") == false
#             push!(anim_traj, "e_mse" => [])
#             push!(anim_traj, "e_pred_ll" => [])
#         end
#         for i=1:n_particles
#             trace = state.traces[i]
#             covariance_fn = get_retval(trace)[1]
#             noise = trace[:noise]
#             mse =  compute_mse(covariance_fn, noise, xs_obs, ys_obs, xs_train, ys_train)
#             pred_ll = predictive_ll(covariance_fn, noise, xs_obs, ys_obs, xs_train, ys_train)
#             e_mse += mse * weights[i]
#             e_pred_ll += pred_ll * weights[i]
#             push!(anim_traj[t], [covariance_fn, noise, weights[i], mse, pred_ll])
#         end
#         # store information gain plotting info for each timestep
#         push!(anim_traj[t], [xs_info_plot, info_plot])
#         # store E[MSE] and E[Predictive Likelihood]
#         println("E[mse]: $e_mse, E[predictive log likelihood]: $e_pred_ll")
#         push!(anim_traj["e_mse"], e_mse)
#         push!(anim_traj["e_pred_ll"], e_pred_ll)
#     end
#
#     # do inference and plot visualization
#     x_obs_traj = Float64[]
#     y_obs_traj = Float64[]
#     @time state = particle_filter_acquisition_AL(xs_train, ys_train, n_particles, pf_callback, anim_traj, x_obs_traj, y_obs_traj, budget, random)
#     make_animation_acquisition(animation_name, anim_traj, n_particles, xs_train, ys_train, xs_test, ys_test, x_obs_traj, y_obs_traj)
#     plot_name = animation_name * "_acc"
#     make_accuracy_plot(plot_name, anim_traj)
#     return anim_traj
# end
#
# dataset_names = ["21", "02", "05", "10", "04"]
# n_particles = 100
# n_trials = 5
#
# for i=1:length(dataset_names)
#     dataset_name = dataset_names[i]
#     n_obs_plotting = haskey(fn_to_obs, dataset_name) ? fn_to_obs[dataset_name] : n_obs_default
#     budget = 20
#
#     anim_traj_random = []
#     anim_traj_AL = []
#
#     for j=1:n_trials
#
#         char = "multi"
#
#         animation_name_rand = char * dataset_name * "_rand_" * string(n_particles) * "_" * string(j)
#         animation_name_al = char * dataset_name * "_active_" * string(n_particles) * "_" * string(j)
#
#         # random
#         push!(anim_traj_random, run_inference(dataset_name, animation_name_rand, n_particles, functions[dataset_name], n_obs_plotting, budget, true))
#         # AL
#         push!(anim_traj_AL, run_inference(dataset_name, animation_name_al, n_particles, functions[dataset_name], n_obs_plotting, budget, false))
#     end
#
#     # plot combined graph
#     plot_name = dataset_name * "_multi_" * string(n_trials) * "_" * string(n_particles)
#     make_acc_plot_multi(plot_name, anim_traj_random, anim_traj_AL)
# end
