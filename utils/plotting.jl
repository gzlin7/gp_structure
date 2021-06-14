using Plots
gr()
Plots.GRBackend()

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
    plot!(plot,pred_xs,pred_ys, linealpha = weight*10, linecolor=:teal,
    ribbon=variances, fillalpha=weight, fillcolor=:lightblue)
end

# plotting functions

function make_animation_sequential(animation_name, anim_traj, n_particles, xs_train, ys_train, xs, ys)
    sorted_obs = []
    for obs in keys(anim_traj)
        push!(sorted_obs, obs)
    end
    anim = @animate for obs in sort!(sorted_obs)
        vals = anim_traj[obs]
        obs_xs = xs_train[1:obs]
        obs_ys = ys_train[1:obs]
        pred_xs = xs

        # plot observations
        p = plot(xs_train, ys_train, title="$obs Observations, $n_particles Particles ", ylim=(minimum(ys_train)-1, maximum(ys_train)+1), legend=false, linecolor=:red)

        # get indices of the top n particles
        # weights = [vals[i][3] for i=1:length(vals)]
        # n = 5
        # best_idxes = get_n_top_weight_idxs(n, weights)

        # plot predictions
        for i=1:length(vals)
            covariance_fn = vals[i][1]
            noise = vals[i][2]
            weight = vals[i][3]
        # plot predictions for top particles
            plot_gp(p, covariance_fn, weight, obs_xs, obs_ys, pred_xs, noise)
            plot!(p, obs_xs, obs_ys, seriestype = :scatter,  marker = (:circle, 3, 0.6, :orange, stroke(1, 1, :black, :dot)))
        end
    end
    gif(anim, "animations/sequential/" * animation_name * ".gif", fps = 1)
end


function make_animation_acquisition(animation_name, anim_traj, n_particles, xs_train, ys_train, xs, ys, x_obs_traj, y_obs_traj)
    sorted_obs = []
    for obs in keys(anim_traj)
        push!(sorted_obs, obs)
    end

    x_min, x_max = minimum(xs_train), maximum(xs_train)
    y_min, y_max = minimum(ys_train), maximum(ys_train)

    sorted_obs = filter!(x->x != "e_mse" && x != "e_pred_ll", sorted_obs)

    anim = @animate for obs in sort!(sorted_obs)
        vals = anim_traj[obs]
        obs_xs = x_obs_traj[1:obs]
        obs_ys = y_obs_traj[1:obs]
        pred_xs = xs

        # information gain
        xs_info_plot, info_plot = vals[length(vals)]
        ys_info_plot = []
        for x in xs_info_plot
            push!(ys_info_plot, ys_train[findfirst(i->i==x, xs_train)])
        end

        # plot observations
        l = @layout [a; b]
        p = plot(xs_train, ys_train, title="$obs Observations, $n_particles Particles ", xlim=(x_min, x_max), ylim=(y_min-1, y_max+1), legend=false, linecolor=:red, layout = l)

        # get indices of the top n particles
        # weights = [vals[i][3] for i=1:length(vals)]
        # n = 5
        # best_idxes = get_n_top_weight_idxs(n, weights)

        # plot predictions
        # println("best_idxes", best_idxes)
        for i=1:length(vals) - 1 # last value stores info gain plot
            covariance_fn = vals[i][1]
            noise = vals[i][2]
            weight = vals[i][3]
            plot_gp(p[1], covariance_fn, weight, obs_xs, obs_ys, pred_xs, noise)
        end

        old_obs = length(obs_xs) - 1
        plot!(p[1], obs_xs[1 : old_obs], obs_ys[1 : old_obs], seriestype = :scatter,  marker = (:circle, 0.4, 8, :yellow))
        plot!(p[1], obs_xs[old_obs+1 : length(obs_xs)], obs_ys[old_obs+1 : length(obs_xs)], seriestype = :scatter,  marker = (:circle, 0.8, 8, :red))

        if length(info_plot) > 0
            max_info_idx = argmax(info_plot)
            low_bound = minimum(info_plot) - 1
            plot!(p[2], obs_xs[1 : old_obs], info_plot[1 : old_obs], seriestype = :scatter,  marker = (:circle, 0.4, 8, :yellow), title="Information Gain", xlim=(x_min, x_max), ylim=(low_bound, maximum(info_plot) + 1), legend=false)
            plot!(p[2], xs_info_plot, info_plot, fillrange=[[low_bound for i=1:length(info_plot)], info_plot], fillalpha=0.6, fillcolor=:green)
            plot!(p[2], [xs_info_plot[max_info_idx]], [info_plot[max_info_idx]],  seriestype = :scatter,  marker = (:star, 0.8, 8, :red))
        end
    end

    gif(anim, "animations/acquisition/" * animation_name * ".gif", fps = 1)
end

function make_accuracy_plot(plot_name, anim_traj)
    e_mse = anim_traj["e_mse"]
    e_pred_ll = anim_traj["e_pred_ll"]
    x = 1:length(e_mse)
    p = plot(x, e_mse, label="e_mse", legend=:bottomright, title=plot_name, xlabel="no. observations")
    plot!(p, x, e_pred_ll, label="e_pred_ll")
    savefig(p, "animations/acquisition/plots/" * plot_name)
end

function make_acc_plot_multi(plot_name, anim_traj_random, anim_traj_AL)
    avg_mse_rand, avg_mse_AL = mean([traj["e_mse"] for traj in anim_traj_random]), mean([traj["e_mse"] for traj in anim_traj_AL])
    avg_predll_rand, avg_predll_AL = mean([traj["e_pred_ll"] for traj in anim_traj_random]), mean([traj["e_pred_ll"] for traj in anim_traj_AL])
    x = 1:length(avg_mse_rand)
    # mse plot
    p_mse = plot(x, avg_mse_rand, label="E[MSE] Random", legend=:right, title=plot_name * "_mse", xlabel="no. observations")
    plot!(p_mse, x, avg_mse_AL, label="E[MSE] Active")
    savefig(p_mse, "animations/acquisition/plots/overall/" * plot_name * "_mse")
    # pred ll plot
    p_mse = plot(x, avg_predll_rand, label="E[Pred LL] Random", legend=:right, title=plot_name * "_pll", xlabel="no. observations")
    plot!(p_mse, x, avg_predll_AL, label="E[Pred LL] Active")
    savefig(p_mse, "animations/acquisition/plots/overall/" * plot_name * "_pll")
end
