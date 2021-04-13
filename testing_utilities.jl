include("utils/shared.jl")
using Plots
gr()
Plots.GRBackend()

function generate_first_layer(n_buckets)
    cov_fns = Dict(Constant => [], Linear => [], SquaredExponential => [], Periodic => [])
    for param1 in LinRange(1/n_buckets,1.0,n_buckets)
        # Constant, Linear, SE
        for kern1 in [Linear, Constant, SquaredExponential]
            push!(cov_fns[kern1], kern1(param1))
        end
        # Periodic- tune second parameter
        for param2 in LinRange(1/n_buckets,1.0,n_buckets)
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
        map_left = node_to_choicemap(node.left, choicemap(),  get_child(depth, 1, 2))
        map_right = node_to_choicemap(node.right, choicemap(), get_child(depth, 2, 2))
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

function run_inference(dataset_name, sequential, cov_fn, xs_train, ys_train)
    anim_traj = Dict()

    # set seed
    Random.seed!(1)

    # do inference and plot visualization
    if (sequential)
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

function make_animation_likelihood(animation_name, results, xs_train, ys_train)
    n_results = length(results)
    anim = @animate for i=1:n_results
        result = results[i]
        cov_fn, likelihood = result
        # plot observations
        rounded_lik = round(likelihood, digits=3)
        p = plot(xs_train, ys_train, title="[$i/$n_results] $cov_fn, likelihood: $rounded_lik", ylim=(-3, 3), legend=false, linecolor=:red)
        plot_gp(p, cov_fn, 0.8, xs_train, ys_train, xs_train)
    end
    gif(anim, "animations/testing/" * animation_name * ".gif", fps = 2)
end

# cov_grid = get_cov_grid(3,2)
# println("COV GRID LENGTH: ", length(cov_grid))
#
# println(cov_grid[7000])
# display(node_to_choicemap(cov_grid[7000]))
