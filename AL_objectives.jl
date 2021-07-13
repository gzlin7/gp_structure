include("utils/shared.jl")
include("utils/unfold_model.jl")
include("utils/plotting.jl")
include("AL_helper.jl")
using Optim
using LinearAlgebra

function plot_metric(metric_fn, intervention_locs)
    # plot sparse grid of information gain
    n_locs = min(30, length(intervention_locs))
    xs_info_plot = intervention_locs[1:length(intervention_locs)÷n_locs:end]
    info_gains_plot = [metric_fn(x) for x in xs_info_plot]
    # info_gains_plot = [get_information_gain(x) for x in xs_info_plot]
    # println("Info Gain " * string(info_gains_plot))

    # argmax_loc = Optim.minimizer(optimize(get_information_gain,  minimum(intervention_locs), maximum(intervention_locs)))
    # return (argmin(abs.(intervention_locs .- argmax_loc)), xs_info_plot, info_gains_plot)
    best_loc = xs_info_plot[argmax(info_gains_plot)]
    best_loc_idx = findfirst(x->x==best_loc, intervention_locs)
    return (best_loc_idx, xs_info_plot, info_gains_plot)
end

# objectives

function random_obj(state, intervention_locs, past_obs_x, past_obs_y)
    return rand(1:length(intervention_locs)), [], []
end

function sequential_obj(state, intervention_locs, past_obs_x, past_obs_y)
    return argmin(intervention_locs), [], []
end

function max_variance(state, intervention_locs, past_obs_x, past_obs_y)
    n_traces = 50
    m = 30

    # sample traces (thetas)
    ret = sample_traces(state, n_traces)
    traces = ret[1]
    weights = ret[2]

    function get_variance(intervention_x)
        total_var = 0
        mu_cov = get_dist_traces(traces, weights, past_obs_x, past_obs_y, intervention_x)
        # for each theta
        for i=1:length(mu_cov)
            (conditional_mu, conditional_cov_matrix) = mu_cov[i]
            mu, var = conditional_mu[1], conditional_cov_matrix[1,1]
            total_var += var
        end
        return total_var
    end

    return plot_metric(get_variance, intervention_locs)
end

function gp_obj(state, intervention_locs, past_obs_x, past_obs_y)
    n_traces = 50
    m = 30

    # sample traces (thetas)
    ret = sample_traces(state, n_traces)
    traces = ret[1]
    weights = ret[2]

    function get_information_gain(intervention_x)
        info_gain = 0
        mu_cov = get_dist_traces(traces, weights, past_obs_x, past_obs_y, intervention_x)

        # for each theta
        for i=1:length(mu_cov)
            (conditional_mu, conditional_cov_matrix) = mu_cov[i]
            mu, var = conditional_mu[1], conditional_cov_matrix[1,1]
            m_outcomes = [normal(mu, var) for t=1:m]
            p_theta = weights[i]

            approx_info_gain = 0
            # for each y
            for y in m_outcomes
                log_py_trace = calc_log_prob_y_given_th(y, mu_cov, i)
                info_gain = log(p_theta) + log_py_trace - calc_log_prob_y(mu_cov, y, weights)
                if info_gain == NaN || info_gain == Inf
                else
                    approx_info_gain += info_gain
                end
            end
            info_gain += p_theta * 1/m * approx_info_gain
        end
        # negative to return max, since minimization
        return info_gain
    end

    return plot_metric(get_information_gain, intervention_locs)
end

function region_of_interest(state, intervention_locs, past_obs_x, past_obs_y)
    n_traces = 50
    m = 30

    # sample traces (thetas)
    ret = sample_traces(state, n_traces)
    traces = ret[1]
    weights = ret[2]

    variances = Float64[]

    # assume all points equally weighted
    function get_mean_marginal_entropy(intervention_x)
        total_var = 0
        mu_cov = get_dist_traces(traces, weights, past_obs_x, past_obs_y, intervention_x)
        # for each theta
        for i=1:length(mu_cov)
            (conditional_mu, conditional_cov_matrix) = mu_cov[i]
            mu, var = conditional_mu[1], conditional_cov_matrix[1,1]
            total_var += log(var)
        end
        return total_var
    end

    return plot_metric(get_mean_marginal_entropy, intervention_locs)
end

function delta_variance(state, intervention_locs, past_obs_x, past_obs_y)
    n_traces = 20
    m = 10

    # sample traces (thetas)
    ret = sample_traces(state, n_traces)
    traces = ret[1]
    weights = ret[2]

    function get_information_gain(intervention_x)
        info_gain = 0
        mu_cov = get_dist_traces(traces, weights, past_obs_x, past_obs_y, intervention_x)

        # for each theta
        for i=1:length(mu_cov)
            (conditional_mu, conditional_cov_matrix) = mu_cov[i]
            mu, var = conditional_mu[1], conditional_cov_matrix[1,1]
            m_outcomes = [normal(mu, var) for t=1:m]
            p_theta = weights[i]

            approx_info_gain = 0
            # for each y
            for y in m_outcomes
                _, cov_matrix_new  = compute_predictive(theta, noise, past_obs_x, past_obs_y, [intervention_x])
                approx_info_gain += det(cov_matrix_new)
            end
            info_gain += approx_info_gain
        end
        # negative to return max, since minimization
        return info_gain
    end

    return plot_metric(get_information_gain, intervention_locs)
end
