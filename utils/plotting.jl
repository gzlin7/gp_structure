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

    anim = @animate for obs in sort!(sorted_obs)
        vals = anim_traj[obs]
        obs_xs = x_obs_traj[1:obs]
        obs_ys = y_obs_traj[1:obs]
        pred_xs = xs

        e_ucb_xs = Array(LinRange(0.0, maximum(xs_train), 50))
        e_ucb_vars = zeros(Float64, length(e_ucb_xs))
        e_ucb_mus =  zeros(Float64, length(e_ucb_xs))
        k = 0.8

        # plot observations
        p = plot(xs_train, ys_train, title="$obs Observations, $n_particles Particles ", xlim= (minimum(xs_train), maximum(xs_train)), ylim=(minimum(ys_train)-1, maximum(ys_train)+1), legend=false, linecolor=:red)

        # get indices of the top n particles
        # weights = [vals[i][3] for i=1:length(vals)]
        # n = 5
        # best_idxes = get_n_top_weight_idxs(n, weights)

        # plot predictions
        # println("best_idxes", best_idxes)
        for i=1:length(vals)
            covariance_fn = vals[i][1]
            noise = vals[i][2]
            weight = vals[i][3]
            plot_gp(p, covariance_fn, weight, obs_xs, obs_ys, pred_xs, noise)

            # add E[UCB] * weight
            (conditional_mu, conditional_cov_matrix) = compute_predictive(
                covariance_fn, noise, obs_xs, obs_ys, e_ucb_xs)

            for j=1:length(e_ucb_xs)
                mu, var = conditional_mu[j], conditional_cov_matrix[j,j]
                e_ucb_vars[j] += (k * var) * weight
                e_ucb_mus[j] += mu * weight
            end
        end
        plot!(p, e_ucb_xs, e_ucb_mus, yerror=e_ucb_vars, alpha=0.5)

        # plot max UCB in diff color
        min_ucb = argmin(e_ucb_mus + e_ucb_vars)
        min_ucb_x = convert(Array{Float64}, [e_ucb_xs[min_ucb]])
        min_ucb_mu = convert(Array{Float64}, [e_ucb_mus[min_ucb]])
        min_ucb_var = convert(Array{Float64}, [e_ucb_vars[min_ucb]])

        old_obs = length(obs_xs) - 1
        plot!(p, obs_xs[1 : old_obs], obs_ys[1 : old_obs], seriestype = :scatter,  marker = (:circle, 0.4, 8, :yellow))
        plot!(p, obs_xs[old_obs+1 : length(obs_xs)], obs_ys[old_obs+1 : length(obs_xs)], seriestype = :scatter,  marker = (:circle, 0.8, 8, :red))
        plot!(p, min_ucb_x, min_ucb_mu, yerror=min_ucb_var, alpha=1, color=:red, markerstrokecolor=:red,  seriestype = :scatter,  marker = (:star5, 0.4, 8, :red))

    end

    gif(anim, "animations/acquisition/" * animation_name * ".gif", fps = 1)
end
