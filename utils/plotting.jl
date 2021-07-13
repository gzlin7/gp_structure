using Plots
include("shared.jl")
gr()
Plots.GRBackend()
#
function get_n_top_weight_idxs(n, weights)
    w = deepcopy(weights)
    best_idxes = []
    for p=1:n
        max_weight_idx = findmax(w)[2]
        push!(best_idxes, max_weight_idx)
        w[max_weight_idx] = 0
    end
    return best_idxes
end
#
function plot_gp(plot, covariance_fn, weight, obs_xs, obs_ys, pred_xs, noise)
    # plot posterior means and vanriance given one covariance fn
    (conditional_mu, conditional_cov_matrix) = compute_predictive(
        covariance_fn, noise, obs_xs, obs_ys, pred_xs)
    variances = []
    for j=1:length(pred_xs)
        mu, var = conditional_mu[j], conditional_cov_matrix[j,j]
        push!(variances, sqrt(var))
    end
    pred_ys = conditional_mu
    plot!(plot,pred_xs, pred_ys, linealpha = weight*10, linecolor=:teal,
    ribbon=variances, fillalpha=weight, fillcolor=:lightblue)
end

# # plotting functions
#
# function make_animation_sequential(animation_name, anim_traj, n_particles, xs_train, ys_train, xs, ys)
#     sorted_obs = []
#     for obs in keys(anim_traj)
#         push!(sorted_obs, obs)
#     end
#     anim = @animate for obs in sort!(sorted_obs)
#         vals = anim_traj[obs]
#         obs_xs = xs_train[1:obs]
#         obs_ys = ys_train[1:obs]
#         pred_xs = xs
#
#         # plot observations
#         p = plot(xs_train, ys_train, title="$obs Observations, $n_particles Particles ", ylim=(minimum(ys_train)-1, maximum(ys_train)+1), legend=false, linecolor=:red)
#
#         # get indices of the top n particles
#         # weights = [vals[i][3] for i=1:length(vals)]
#         # n = 5
#         # best_idxes = get_n_top_weight_idxs(n, weights)
#
#         # plot predictions
#         for i=1:length(vals)
#             covariance_fn = vals[i][1]
#             noise = vals[i][2]
#             weight = vals[i][3]
#         # plot predictions for top particles
#             plot_gp(p, covariance_fn, weight, obs_xs, obs_ys, pred_xs, noise)
#             plot!(p, obs_xs, obs_ys, seriestype = :scatter,  marker = (:circle, 3, 0.6, :orange, stroke(1, 1, :black, :dot)))
#         end
#     end
#     gif(anim, "animations/sequential/" * animation_name * ".gif", fps = 1)
# end

function make_plot_acquisition(obs, all_obs, inference_ret, objective, info_metric, bounds)
    x_min, x_max, y_min, y_max, info_metric_min, info_metric_max = bounds
    anim_traj, n_particles, xs_train, ys_train, x_obs_traj, y_obs_traj = inference_ret

    vals = anim_traj[obs]
    obs_xs = x_obs_traj[1:obs-1]
    obs_ys = y_obs_traj[1:obs-1]
    pred_xs = xs_train

    # information gain
    xs_info_plot, info_plot = last(vals)
    ys_info_plot = []
    for x in xs_info_plot
        push!(ys_info_plot, ys_train[findfirst(i->i==x, xs_train)])
    end

    # plot observations
    l = @layout [a; b]
    p = plot(xs_train, ys_train, title="$objective, $(obs-1) obs", xlim=(x_min, x_max), ylim=(y_min-1, y_max+1), legend=false, linecolor=:red, layout = l)

    for i=1:length(vals) - 1 # last value stores info gain plot
        covariance_fn = vals[i][1]
        noise = vals[i][2]
        weight = vals[i][3]
        plot_gp(p[1], covariance_fn, weight, obs_xs, obs_ys, pred_xs, noise)
    end

    old_obs = length(obs_xs) - 1
    plot!(p[1], obs_xs[1 : old_obs], obs_ys[1 : old_obs], seriestype = :scatter,  marker = (:circle, 0.4, 8, :yellow))
    plot!(p[1], obs_xs[old_obs+1 : length(obs_xs)], obs_ys[old_obs+1 : length(obs_xs)], seriestype = :scatter,  marker = (:circle, 0.8, 8, :red))

    if info_metric != " "
        max_info_idx = argmax(info_plot)
        plot!(p[2], obs_xs[1 : old_obs], info_plot[1 : old_obs], seriestype = :scatter,  marker = (:circle, 0.4, 8, :yellow), title=info_metric, xlim=(x_min, x_max), ylim=(info_metric_min, info_metric_max), legend=false)
        plot!(p[2], xs_info_plot, info_plot, fillalpha=0.6, fillcolor=:green,  fillrange=[info_metric_min for i=1:length(info_plot)])
        plot!(p[2], [xs_info_plot[max_info_idx]], [info_plot[max_info_idx]],  seriestype = :scatter,  marker = (:star, 0.8, 8, :red))
    end
    return p
end

function make_accuracy_plot(plot_title, anim_traj, bounds)
    e_mse = anim_traj["e_mse"]
    e_pred_ll = anim_traj["e_pred_ll"]
    x = 1:length(e_mse)
    p = plot(x, e_mse, title=plot_title, label="e_mse", legend=:bottomright, xlabel="no. observations", ylim = bounds)
    plot!(p, x, e_pred_ll, label="e_pred_ll")
    return p
end

function make_animation_acquisition_multi(animation_name, obj_to_info_metric, inference_ret_all)
    objectives = collect(keys(inference_ret_all))
    n_trials = length(inference_ret_all[objectives[1]])
    first_inference_ret = inference_ret_all[objectives[1]][1]
    anim_traj, n_particles, xs_train, ys_train, x_obs_traj, y_obs_traj = first_inference_ret
    all_obs = collect(keys(anim_traj))
    all_obs = filter!(x->x != "e_mse" && x != "e_pred_ll", all_obs)
    sort!(all_obs)

    x_min, x_max = minimum(xs_train), maximum(xs_train)
    y_min, y_max = minimum(ys_train), maximum(ys_train)
    plot_bounds = [x_min, x_max, y_min, y_max]

    # for each trial
    for i=1:n_trials
        # get info metric plot boudnds
        objective_info_bounds = Dict()
        for objective in objectives
            if objective in keys(obj_to_info_metric)
                anim_traj = inference_ret_all[objective][i][1]
                all_info_metric_vals = vcat([anim_traj[obs][length(anim_traj[obs])][2] for obs in all_obs]...)
                filter!(x->!isnan(x), all_info_metric_vals)
                objective_info_bounds[objective] = [minimum(all_info_metric_vals), maximum(all_info_metric_vals)]
            else
                objective_info_bounds[objective] = [0, 100]
            end
        end

        # plot all objectives
        anim = @animate for obs in all_obs
            plots = []
            for objective in objectives
                    bounds = vcat(plot_bounds, objective_info_bounds[objective])
                    inference_ret = inference_ret_all[objective][i]
                    info_metric = haskey(obj_to_info_metric, objective) ? obj_to_info_metric[objective] : " "
                    push!(plots, make_plot_acquisition(obs, all_obs, inference_ret, objective, info_metric, bounds))
            end
            p = plot(plots...,  size = (1500, 700), layout = @layout [a b c; d{0.33w} e{0.33w}])
        end
        # animate inference
        gif(anim, animation_name * "_" * string(i) * ".gif", fps = 1)

        # plot accuracy
        anim_trajs = [(objective, inference_ret_all[objective][i][1]) for objective in objectives]
        all_vals = []
        for item in anim_trajs
            objective, anim_traj = item
            push!(all_vals, minimum(anim_traj["e_mse"]))
            push!(all_vals, minimum(anim_traj["e_pred_ll"]))
            push!(all_vals, maximum(anim_traj["e_mse"]))
            push!(all_vals, maximum(anim_traj["e_pred_ll"]))
        end
        acc_bounds = minimum(all_vals), maximum(all_vals)

        acc_plots = []
        for item in anim_trajs
            objective, anim_traj = item
            push!(acc_plots, make_accuracy_plot(objective * " " * string(i), anim_traj, acc_bounds))
        end
        acc_combined = plot(acc_plots...,  size = (1500, 700), layout = @layout [a b c; d{0.33w} e{0.33w}])
        savefig(acc_combined,  animation_name * "_" * string(i) * "_acc")
    end
end

function make_acc_plot_multi(plot_name, inference_ret_all)
    mse_plot = plot(title=plot_name * " E[MSE]", xlabel="no. observations", legend=false)
    pll_plot = plot(title=plot_name * " E[Pred LL]", xlabel="no. observations", legend=:bottomright)

    for (obj, rets) in inference_ret_all
        trajectories = [ret[1] for ret in rets]
        avg_mse = mean([traj["e_mse"] for traj in trajectories])
        avg_predll = mean([traj["e_pred_ll"] for traj in trajectories])
        x = 1:length(avg_mse)

        # mse plot
        plot!(mse_plot, x, avg_mse, label=obj)
        # pred ll plot
        plot!(pll_plot, x, avg_predll, label=obj)
    end
    combined_plot = plot(mse_plot, pll_plot, layout = (1,2), size = (900, 300))
    savefig(combined_plot, "animations/acquisition/plots/" * plot_name)
end
