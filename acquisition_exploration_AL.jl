include("utils/shared.jl")
include("utils/unfold_model.jl")
include("utils/plotting.jl")
using GenParticleFilters
using Optim
using StatsBase
@load_generated_functions()

# acquisition functions

# UCB
function ucb_fn(mu, var)
    k = 0.8
    return mu + k * var
end

# EI http://krasserm.github.io/2018/03/21/bayesian-optimization/
# function ei_fn(mu,var)
#
# end


function particle_filter_acquisition_AL(xs::Vector{Float64}, ys::Vector{Float64}, n_particles, callback, anim_traj, x_obs_traj, y_obs_traj, budget, random::Bool=false)
    # n_obs = length(xs)
    n_explore = 0
    obs_idx = [1]
    obs_xs = [xs[1]]
    obs_ys = [ys[1]]
    obs_choices = [choicemap((:state => 1 => :x, xs[1]), (:state => 1 => :y, ys[1]))]
    # keep track of xs we have not evaluated yet
    potential_xs = deepcopy(xs)
    deleteat!(potential_xs, 1)

    state = pf_initialize(model, (1,), obs_choices[1], n_particles)

    push!(x_obs_traj, xs[1])
    push!(y_obs_traj, ys[1])


    # Iterate across timesteps
    for t=2:budget-1
        # Resample and rejuvenate if the effective sample size is too low
        # if effective_sample_size(state) < 0.5 * n_particles
        # Perform residual resampling, pruning low-weight particles
        if (mod(t,10) == 0)
            pf_resample!(state, :multinomial)
            # Perform a rejuvenation move on past choices
            pf_rejuvenate!(state, mh, (subtree_proposal, (), subtree_involution))
            pf_rejuvenate!(state, mh, (noise_proposal, ()))
        end

        # select next observation point
        if (t > n_explore) && !random
            potential_xs_idx = get_next_obs_x(state, potential_xs, obs_xs, obs_ys)
        else
            potential_xs_idx = rand(1:length(potential_xs))
        end
        next_x = potential_xs[potential_xs_idx]
        deleteat!(potential_xs, potential_xs_idx)

        next_obs_idx = findfirst(isequal(next_x), xs)
        push!(obs_idx, next_obs_idx)
        push!(obs_xs, xs[next_obs_idx])
        push!(obs_ys, ys[next_obs_idx])

        # Update filter state with new observation at timestep t
        push!(obs_choices, choicemap((:state => t => :x, xs[next_obs_idx]), (:state => t => :y, ys[next_obs_idx])))
        pf_update!(state, (t,), (UnknownChange(),), obs_choices[t])
        push!(x_obs_traj, xs[next_obs_idx])
        push!(y_obs_traj, ys[next_obs_idx])
        if mod(t,5) == 0
            println(obs_idx)
            println("number of observations: $t")
        end
        callback(state, xs, ys, anim_traj, t)
    end
    return state
end

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

        (conditional_mu, conditional_cov_matrix) = compute_predictive(
            theta, noise, past_obs_x, past_obs_y, [intervention_x])
        # mu, var = conditional_mu[1], conditional_cov_matrix[1,1]
        push!(mu_cov, (conditional_mu, conditional_cov_matrix))
    end
    return mu_cov
end

# P(Y|theta)
function calc_prob_y_given_th(y, mu_cov, theta_idx)
    (conditional_mu, conditional_cov_matrix) = mu_cov[theta_idx]
    return exp(logpdf(mvnormal, [y], conditional_mu, conditional_cov_matrix))
end

# P(Y) over all thetas
function calc_prob_y(mu_cov, y)
    p_y = 0
    for i=1:length(mu_cov)
        p_y += calc_prob_y_given_th(y, mu_cov, i)
    end
    return p_y
end

function get_next_obs_x(state, intervention_locs, past_obs_x, past_obs_y)
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
                py_trace = calc_prob_y_given_th(y, mu_cov, i)
                approx_info_gain += log(py_trace / calc_prob_y(mu_cov, y))
            end
            info_gain += p_theta * 1/m * approx_info_gain
        end
        # negative to return max, since minimization
        return -info_gain
    end

    argmax_loc = Optim.minimizer(optimize(get_information_gain,  minimum(intervention_locs), maximum(intervention_locs)))
    return argmin(abs.(intervention_locs .- argmax_loc))
end
