include("utils/shared.jl")
using StatsBase

# return n_samples (sample unweighted) from state, with corresponding normalized weight array
function sample_traces(state, n_samples)
    state_weights = get_norm_weights(state)
    state_traces = state.traces
    # sample n_samples traces without replacement
    sample_idxes = StatsBase.knuths_sample!([i for i=1:length(state_traces)], [1 for i=1:n_samples])
    # get weights and traces
    traces = []
    weights = []
    sum_weights = 0
    for i in sample_idxes
        push!(traces, state_traces[i])
        weight = state_weights[i]
        sum_weights += weight
        push!(weights, weight)
    end
    # renormalize weights
    weights = weights / sum_weights
    return (traces, weights)
end

# return array of (conditional_mu, conditional_cov_matrix) for each covariance_fn in trace (in order)
function get_dist_traces(traces, weights, past_obs_x, past_obs_y, intervention_x)
    mu_cov = []
    for i=1:length(traces)
        trace = traces[i]
        theta = get_retval(trace)[1]
        noise = trace[:noise]
        weight = weights[i]

        conditional_mu, conditional_cov_matrix = (length(past_obs_x) == 0) ?
        (zeros(1), compute_cov_matrix(theta, noise, [intervention_x])) :
        compute_predictive(theta, noise, past_obs_x, past_obs_y, [intervention_x])

        push!(mu_cov, (conditional_mu, conditional_cov_matrix))
    end
    return mu_cov
end

# P(Y|theta)
function calc_log_prob_y_given_th(y, mu_cov, theta_idx)
    (conditional_mu, conditional_cov_matrix) = mu_cov[theta_idx]
    return logpdf(mvnormal, [y], conditional_mu, conditional_cov_matrix)
end

# P(Y) over all thetas
function calc_log_prob_y(mu_cov, y, weights)
    probs = [calc_log_prob_y_given_th(y, mu_cov, i) + log(weights[i]) for i=1:length(mu_cov)]
    return logsumexp(probs)
end
