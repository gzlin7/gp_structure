include("sequential.jl")
include("acquisition_exploration.jl")
include("utils/shared.jl")

function generate_first_layer(n_buckets)
    cov_fns = []
    for param1 in LinRange(1/n_buckets,1.0,n_buckets)
        # Constant, Linear, SE
        for kern1 in [Linear, Constant, SquaredExponential]
            push!(cov_fns, kern1(param1))
        end
        # Periodic- tune second parameter
        for param2 in LinRange(1/n_buckets,1.0,n_buckets)
            push!(cov_fns, Periodic(param1,param2))
        end
    end
    return cov_fns
end

function add_tree_layer(cov_grid)
    new_cov_fns = []
    for i=1:length(cov_grid)-1
        for j=i+1:length(cov_grid)
            for operator in [Plus, Times]
                push!(new_cov_fns, operator(cov_grid[i], cov_grid[j]))
            end
        end
    end
    return new_cov_fns
end

# only works up to depth 2 trees for now
node_types = [Constant, Linear, SquaredExponential, Periodic, Plus, Times]
function node_to_choicemap(node, map, depth)
    type = findfirst(x->x==typeof(node), node_types)
    map[(depth, :type)] = type
    # Plus, Times (recurse)
    if type in [5,6]
        map = node_to_choicemap(node.left, map, depth+1)
        return node_to_choicemap(node.right, map, depth+2)
    else
        # Constant, Linear
        if type in [1,2]
            map[(depth, :param)] = node.param
        # Squared Exponential
        elseif type == 3
            map[(depth, :param)] = node.length_scale
        # Periodic
        elseif type == 4
            map[(depth, :scale)] = node.scale
            map[(depth, :period)] = node.period
        end
        return map
    end
end

function get_covariance_grid(tree_depth, n_buckets)
    cov_grid = generate_first_layer(n_buckets)
    prev_grid = cov_grid
    for n=2:tree_depth
        new_layer = add_tree_layer(prev_grid)
        prev_grid = new_layer
        cov_grid = [cov_grid; new_layer]
    end
    return cov_grid
end

cov_grid = get_covariance_grid(2,5)
cov_grid_choicemaps = [node_to_choicemap(cov_grid[t], choicemap(), 1) for t=1:length(cov_grid)]
println(length(cov_grid))
println(length(cov_grid_choicemaps))
println(cov_grid[100])
println(display(cov_grid_choicemaps[100]))

# function run_inference(dataset_name, animation_name, n_particles, sequential)
#     # load the dataset
#     if (dataset_name == "airline")
#         (xs, ys) = get_airline_dataset()
#     else
#         (xs, ys) = get_dataset(dataset_name)
#     end
#     xs_train = xs[1:100]
#     ys_train = ys[1:100]
#     xs_test = xs[101:end]
#     ys_test = ys[101:end]
#
#     anim_traj = Dict()
#
#     # set seed
#     Random.seed!(1)
#
#     function pf_callback(state, xs_train, ys_train, anim_traj, t)
#         # calculate E[MSE]
#         n_particles = length(state.traces)
#         e_mse = 0
#         e_pred_ll = 0
#         weights = get_norm_weights(state)
#         if haskey(anim_traj, t) == false
#             push!(anim_traj, t => [])
#         end
#         for i=1:n_particles
#             trace = state.traces[i]
#             covariance_fn = get_retval(trace)[1]
#             noise = trace[:noise]
#             push!(anim_traj[t], [covariance_fn, noise, weights[i]])
#             mse =  compute_mse(covariance_fn, noise, xs_train, ys_train, xs_test, ys_test)
#             pred_ll = predictive_ll(covariance_fn, noise, xs_train, ys_train, xs_test, ys_test)
#             e_mse += mse * weights[i]
#             e_pred_ll += pred_ll * weights[i]
#         end
#         println("E[mse]: $e_mse, E[predictive log likelihood]: $e_pred_ll")
#     end
#
#     # do inference and plot visualization
#     if (sequential)
#         @time state = particle_filter_sequential(xs_train, ys_train, n_particles, pf_callback, anim_traj)
#         make_animation_sequential(animation_name, anim_traj, n_particles, xs_train, ys_train, xs, ys)
#     else
#         x_obs_traj = Float64[]
#         y_obs_traj = Float64[]
#         @time state = particle_filter_acquisition(xs_train, ys_train, n_particles, pf_callback, anim_traj, x_obs_traj, y_obs_traj)
#         make_animation_acquisition(animation_name, anim_traj, n_particles, xs_train, ys_train, xs, ys, x_obs_traj, y_obs_traj)
#     end
# end
#
#
# # dataset_names = ["airline"]
# # dataset_names = ["quadratic", "changepoint", "polynomial"]
# dataset_names = ["sinusoid"]
#
#
# for i=1:length(dataset_names)
#     dataset_name = dataset_names[i]
#
#     # # run sequential prediction
#     n_particles = 100
#     sequential = true
#     animation_name = "sequential_" * dataset_name
#     run_inference(dataset_name, animation_name, n_particles, sequential)
#
#     # run acquisition prediction
#     # n_particles = 1
#     # sequential = false
#     # animation_name = "acq_exp_" * dataset_name
#     # # animation_name = "acquisition_" * dataset_name
#     # run_inference(dataset_name, animation_name, n_particles, sequential)
# end
