using Plots
gr()
Plots.GRBackend()

function plot_gp(plot, covariance_fn, noise, weight, obs_xs, obs_ys, pred_xs)
    (conditional_mu, conditional_cov_matrix) = compute_predictive(
        covariance_fn, 0.001, obs_xs, obs_ys, pred_xs)
    variances = []
    for j=1:length(pred_xs)
        mu, var = conditional_mu[j], conditional_cov_matrix[j,j]
        push!(variances, sqrt(var))
    end
    pred_ys = mvnormal(conditional_mu, conditional_cov_matrix)
    plot!(plot,pred_xs,pred_ys, linealpha = max(0.3, weight*10), linecolor=:teal,
    ribbon=variances, fillalpha=max(0.15, weight*5), fillcolor=:lightblue)
end

function make_animation_sequential(animation_name, anim_traj, n_particles, xs_train, ys_train, xs, ys)
    sorted_obs = []
    for obs in keys(anim_traj)
        push!(sorted_obs, obs)
    end
    anim = @animate for obs in sort!(sorted_obs)
        vals = anim_traj[obs]
        obs_xs = xs_train[1:obs]
        obs_ys = ys_train[1:obs]
        pred_xs = xs[obs+1:length(xs)]

        # inter_obs_x = Array{Float64,1}([obs_xs[1]])
        # inter_obs_y = Array{Float64,1}([obs_ys[1]])
        obs_variances = Array{Float64,1}([0])
        for j=2:length(obs_xs)
        #     push!(inter_obs_x, (obs_xs[j]+obs_xs[j-1])/2)
        #     push!(inter_obs_y, (obs_ys[j]+obs_ys[j-1])/2)
            push!(obs_variances, 0)
        end

        # plot observations
        p = plot(xs_train, ys_train, title="$obs Observations, $n_particles Particles ", ylim=(-2, 3), legend=false, linecolor=:red)

        # get indices of the top n particles
        weights = [vals[i][3] for i=1:length(vals)]
        best_idxes = []
        for p=1:5
            max_weight_idx = findmax(weights)[2]
            push!(best_idxes, max_weight_idx)
            weights[max_weight_idx] = 0
        end

        # plot predictions
        for i=1:length(vals)
            covariance_fn = vals[i][1]
            noise = vals[i][2]
            weight = vals[i][3]
            # plot predictions for top particles
            if i in best_idxes
                plot_gp(p, covariance_fn, noise, weight, obs_xs, obs_ys, pred_xs)
            end
            # plot variance on observed data
            (obs_conditional_mu, obs_conditional_cov_matrix) = compute_predictive(
                covariance_fn, 0.001, obs_xs, obs_ys, obs_xs)
            for j=1:length(obs_xs)
                mu, var = obs_conditional_mu[j], obs_conditional_cov_matrix[j,j]
                obs_variances[j] += sqrt(var)
            end
            plot!(p, obs_xs, obs_ys, ribbon=obs_variances, fillalpha=weight*8)
        end
        plot!(p, obs_xs, obs_ys, seriestype = :scatter,  marker = (:circle, 3, 0.6, :orange, stroke(1, 1, :black, :dot)))
    end

    gif(anim, "animations/" * animation_name * ".gif", fps = 1)
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

        inter_obs_x = Array{Float64,1}([obs_xs[1]])
        inter_obs_y = Array{Float64,1}([obs_ys[1]])
        obs_variances = Array{Float64,1}([0])
        for j=2:length(obs_xs)
            push!(inter_obs_x, (obs_xs[j]+obs_xs[j-1])/2)
            push!(inter_obs_y, (obs_ys[j]+obs_ys[j-1])/2)
            push!(obs_variances, 0)
        end

        # plot observations
        p = plot(xs_train, ys_train, title="$obs Observations, $n_particles Particles ", ylim=(-2, 3), legend=false, linecolor=:red)

        # get indices of the top n particles
        weights = [vals[i][3] for i=1:length(vals)]
        best_idxes = []
        for p=1:5
            max_weight_idx = findmax(weights)[2]
            push!(best_idxes, max_weight_idx)
            weights[max_weight_idx] = 0
        end

        # plot predictions
        for i=1:length(vals)
            covariance_fn = vals[i][1]
            noise = vals[i][2]
            weight = vals[i][3]
            # calculate variance on observed data
            (obs_conditional_mu, obs_conditional_cov_matrix) = compute_predictive(
                covariance_fn, noise, obs_xs, obs_ys, inter_obs_x)
            for j=1:length(inter_obs_x)
                mu, var = obs_conditional_mu[j], obs_conditional_cov_matrix[j,j]
                obs_variances[j] += sqrt(var)/mu * weight
            end
            # plot predictions for top particles
            if i in best_idxes
                plot_gp(p, covariance_fn, noise, weight, obs_xs, obs_ys, pred_xs)
            end
        end
        # plot!(p, inter_obs_x, inter_obs_y, ribbon=obs_variances,  fillalpha=0.3)
        plot!(p, obs_xs, obs_ys, seriestype = :scatter,  marker = (:circle, 0.6, 8, :yellow))
    end

    gif(anim, "animations/" * animation_name * ".gif", fps = 1)
end
