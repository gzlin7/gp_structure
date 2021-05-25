include("utils/shared.jl")
include("utils/unfold_model.jl")
include("utils/plotting.jl")
using GenParticleFilters
using Optim
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

function get_next_obs_x(state, intervention_locs, past_obs_x, past_obs_y)
    n_traces = 50
    m = 50
    weights = get_norm_weights(state)

    # Continuous: use Optim to find next x
    function get_information_gain(intervention_x)
        # TODO: verify unweighted trace sampling or switch to weighting
        traces = sample_unweighted_traces(state, n_traces)
        info_gain = 0

        # for g in G
        for i=1:n_traces
            trace = traces[i]
            covariance_fn = get_retval(trace)[1]
            noise = trace[:noise]
            weight = weights[i]

            g = covariance_fn
            P_g = weight

            # TODO: integrate over outcomes using MC sampling
            # for now just assume BEST y?
            (conditional_mu, conditional_cov_matrix) = compute_predictive(
                covariance_fn, noise, past_obs_x, past_obs_y, [intervention_x])
            mu, var = conditional_mu[1], conditional_cov_matrix[1,1]
            m_outcomes = [normal(mu, var) for t=1:m]

            approx_info_gain = 0
            for outcome in m_outcomes
                log_Py_g = predictive_ll(g, noise, past_obs_x, past_obs_y, [intervention_x], [outcome])
                approx_info_gain += (log_Py_g + log(P_g))
            end
            info_gain += 1/m * approx_info_gain
        end
        # negative to return max, since minimization
        return -info_gain
    end

    argmax_loc = Optim.minimizer(optimize(get_information_gain,  minimum(intervention_locs), maximum(intervention_locs)))
    return argmin(abs.(intervention_locs .- argmax_loc))
end
