include("utils/shared.jl")
include("utils/unfold_model.jl")
include("utils/plotting.jl")
using GenParticleFilters
using Plots
using Optim
gr()
Plots.GRBackend()
@load_generated_functions()

dataset_name = "cubic"
animation_name = "acquisition_" * dataset_name
n_particles = 50

function particle_filter(xs::Vector{Float64}, ys::Vector{Float64}, n_particles, callback, anim_traj, x_obs_traj, y_obs_traj)
    # n_obs = length(xs)
    n_obs = 50
    obs_idx = [1]
    obs_xs = [xs[1]]
    obs_ys = [ys[1]]
    obs_choices = [choicemap((:state => 1 => :x, xs[1]), (:state => 1 => :y, ys[1]))]
    # keep track of xs we have not evaluated yet
    potential_xs = deepcopy(xs)
    deleteat!(potential_xs, 1)

    state = pf_initialize(model, (1,), obs_choices[1], n_particles)

    # Iterate across timesteps
    for t=2:n_obs-1
        # Resample and rejuvenate if the effective sample size is too low
        if effective_sample_size(state) < 0.5 * n_particles
            # Perform residual resampling, pruning low-weight particles
            pf_resample!(state, :residual)
            # Perform a rejuvenation move on past choices
            pf_rejuvenate!(state, mh, (subtree_proposal, (), subtree_involution))
            pf_rejuvenate!(state, mh, (noise_proposal, ()))
        end

        # select next observation point
        potential_xs_idx = get_next_obs_x(state, potential_xs, obs_xs, obs_ys)
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
            callback(state, xs, ys, anim_traj, t)
        end
    end
    return state
end

function get_next_obs_x(state, new_xs, x_obs, y_obs)
    k = 0.8
    e_ucb = zeros(Float64, length(new_xs))
    weights = get_norm_weights(state)

    # Continuous: use Optim to find next x
    function get_e_ucb(x)
        e_ucb1 = 0
        for i=1:n_particles
            trace = state.traces[i]
            covariance_fn = get_retval(trace)[1]
            noise = trace[:noise]
            (conditional_mu, conditional_cov_matrix) = compute_predictive(
                covariance_fn, noise, x_obs, y_obs, [x])

            mu, var = conditional_mu[1], conditional_cov_matrix[1,1]
            e_ucb1 += (mu + k * var) * weights[i]
        end
        # negative to return max, since minimization
        return -e_ucb1
    end

    x_maximizer = Optim.minimizer(optimize(get_e_ucb,  0.0, 1.0))
    println("maximizer = ", x_maximizer)
    # return argmin(abs.(new_xs .- x_minimizer))

    # discrete: iteratively find next x
    for i=1:n_particles
        trace = state.traces[i]
        covariance_fn = get_retval(trace)[1]
        noise = trace[:noise]
        (conditional_mu, conditional_cov_matrix) = compute_predictive(
            covariance_fn, noise, x_obs, y_obs, new_xs)

        for j=1:length(new_xs)
            mu, var = conditional_mu[j], conditional_cov_matrix[j,j]
            # print("mu, var")
            # println(mu)
            # println(var)
            e_ucb[j] += (mu + k * var) * weights[i]
        end
    end
    return argmax(e_ucb)
end

# load the dataset
(xs, ys) = get_dataset(dataset_name)
xs_train = xs[1:100]
ys_train = ys[1:100]
xs_test = xs[101:end]
ys_test = ys[101:end]

# visualization
anim_traj = Dict()

# set seed
Random.seed!(1)

pf_callback = (state, xs, ys, anim_traj, t) -> begin
    # calculate E[MSE]
    n_particles = length(state.traces)
    e_mse = 0
    e_pred_ll = 0
    weights = get_norm_weights(state)
    if haskey(anim_traj, t) == false
        push!(anim_traj, t => [])
    end
    for i=1:n_particles
        trace = state.traces[i]
        covariance_fn = get_retval(trace)[1]
        noise = trace[:noise]
        push!(anim_traj[t], [covariance_fn, noise, weights[i]])
        mse =  compute_mse(covariance_fn, noise, xs_train, ys_train, xs_test, ys_test)
        pred_ll = predictive_ll(covariance_fn, noise, xs_train, ys_train, xs_test, ys_test)
        e_mse += mse * weights[i]
        e_pred_ll += pred_ll * weights[i]
    end
    println("E[mse]: $e_mse, E[predictive log likelihood]: $e_pred_ll")
end

x_obs_traj = Float64[]
y_obs_traj = Float64[]
state = particle_filter(xs_train, ys_train, n_particles, pf_callback, anim_traj, x_obs_traj, y_obs_traj)

make_animation_acquisition(animation_name, anim_traj)
