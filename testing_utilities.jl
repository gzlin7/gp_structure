include("utils/shared.jl")
include("slow_sequential.jl")
using Plots
gr()
Plots.GRBackend()

function generate_first_layer(n_buckets)
    cov_fns = Dict(Constant => [], Linear => [], SquaredExponential => [], Periodic => [])
    for param1 in LinRange(1/n_buckets,1.0,n_buckets)
        param1 = round(param1, digits=2)
        # Constant, Linear, SE
        for kern1 in [Linear, Constant, SquaredExponential]
            push!(cov_fns[kern1], kern1(param1))
        end
        # Periodic- tune second parameter
        for param2 in LinRange(1/n_buckets,1.0,n_buckets)
            param2 = round(param2, digits=2)
            push!(cov_fns[Periodic],  Periodic(param1,param2))
        end
    end
    sorted_cov_fns = []
    for (key, value) in cov_fns
        sorted_cov_fns = [sorted_cov_fns; value]
    end
    return sorted_cov_fns
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

node_types = [Constant, Linear, SquaredExponential, Periodic, Plus, Times]
function choicemap_helper(node, map, depth)
    type = findfirst(x->x==typeof(node), node_types)
    map[(depth, :type)] = type
    # Plus, Times (recurse)
    if type in [5,6]
        map_left = choicemap_helper(node.left, choicemap(),  get_child(depth, 1, 2))
        map_right = choicemap_helper(node.right, choicemap(), get_child(depth, 2, 2))
        return merge(map, merge(map_left, map_right))
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

function node_to_choicemap(node)
    choices = choicemap_helper(node, choicemap(), 1)
    tree_choicemap = choicemap()
    set_submap!(tree_choicemap, :tree, choices)
    return tree_choicemap
end

function get_cov_grid(tree_depth, n_buckets)
    cov_grid = generate_first_layer(n_buckets)
    prev_grid = cov_grid
    for n=2:tree_depth
        new_layer = add_tree_layer(prev_grid)
        prev_grid = new_layer
        cov_grid = [cov_grid; new_layer]
    end
    return cov_grid
    # convert to choicemaps
    # return [node_to_choicemap(cov_grid[t], choicemap(), 1) for t=1:length(cov_grid)]
end

function run_inference(dataset_name, sequential, cov_fn, xs_train, ys_train, noise)
    anim_traj = Dict()

    # set seed
    Random.seed!(1)

    # do inference and plot visualization
    if (sequential)
        obs_choices = choicemap()
        for t=1:length(ys_train)
            obs_choices[(:y, t)] = ys_train[t]
        end
        cov_fn_map = node_to_choicemap(cov_fn)
        # display(merge(obs_choices, cov_fn_map))
        # println(merge(obs_choices, cov_fn_map))
        choices = merge(obs_choices, cov_fn_map)
        choices[:noise] = noise
        trace, weight = generate(model, Tuple([xs_train]), choices)
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
    return (cov_fn, likelihood, noise)
end

function plot_covfn(plot, covariance_fn, weight, obs_xs, obs_ys, pred_xs, noise)
    # plot posterior means and vanriance given one covariance fn
    (conditional_mu, conditional_cov_matrix) = compute_predictive(
        covariance_fn, 0.001, obs_xs, obs_ys, pred_xs)
    variances = []
    for j=1:length(pred_xs)
        mu, var = conditional_mu[j], conditional_cov_matrix[j,j]
        push!(variances, sqrt(var))
    end
    pred_ys = mvnormal(conditional_mu, conditional_cov_matrix)
    plot!(plot,pred_xs,pred_ys, linealpha = weight*10, linecolor=:teal,
    ribbon=variances, fillalpha=weight*8, fillcolor=:lightblue,
    legend=true, label = "$covariance_fn")
    plot!(plot,pred_xs,pred_ys, linealpha = weight*10, linecolor=:teal,
    ribbon=variances, fillalpha=weight*8, fillcolor=:green,
    legend=true, label = "noise = $noise")
end

function make_animation_likelihood(animation_name, results, xs_train, ys_train, sumexp, dataset_name)
    n_results = length(results)
    anim = @animate for i=1:n_results
        result = results[i]
        cov_fn, likelihood, noise = result
        # plot observations
        cond_py = round(likelihood/sumexp, digits=3)
        p = plot(xs_train, ys_train, title="[$i/$n_results]", ylim=(-3, 3), linecolor=:red, legend=true, label = "P(Y | cov_fn): $cond_py")
        plot_covfn(p, cov_fn, 0.8, xs_train, ys_train, xs_train, noise)
    end
    gif(anim, "animations/testing/" * dataset_name * "/" * animation_name * "_" * string(length(xs_train)) * ".gif", fps = 2)
end

# cov_grid = get_cov_grid(3,2)
# println("COV GRID LENGTH: ", length(cov_grid))
#
# println(cov_grid[7000])
# display(node_to_choicemap(cov_grid[7000]))
