include("utils/shared.jl")
include("utils/plotting.jl")
using GenParticleFilters

dataset_name = "cubic"
animation_name = "slow_sequential" * dataset_name
n_particles = 50

@gen function model(xs::Vector{Float64})
    n = length(xs)

    # sample covariance function
    covariance_fn::Node = @trace(covariance_prior(1), :tree)

    # sample diagonal noise
    noise = @trace(gamma(1, 1), :noise) + 0.01

    # compute covariance matrix
    cov_matrix = compute_cov_matrix_vectorized(covariance_fn, noise, xs)

    # sample from multivariate normal
    ys = Float64[]
    mu = 0
    var = cov_matrix[1,1]
    covm_22_inv = [1/cov_matrix[1,1]]
    for (i,x) in enumerate(xs)
        # condition on previous ys
        if i > 1
            covm_11 = cov_matrix[i,i]
            covm_21 = cov_matrix[1:i-1,i]
            covm_12 = transpose(covm_21)
            if i > 2
                covm_22_inv = blockwise_inv(cov_matrix, covm_22_inv, i)
            end
            mu = (covm_12 * covm_22_inv * ys)[1]
            var = covm_11[1] - (covm_12 * covm_22_inv * covm_21)[1]
        end
        y = {(:y, i)} ~ normal(mu, sqrt(var))
        push!(ys, y)
    end
    return (covariance_fn, ys)
end

function particle_filter(xs::Vector{Float64}, ys::Vector{Float64}, n_particles, callback, anim_traj, xs_test, ys_test)
    n_obs = length(xs)
    obs_choices = [choicemap(((:y, t), ys[t])) for t=1:n_obs]
    state = pf_initialize(model, ([xs[1]],), obs_choices[1], n_particles)
    # Iterate across timesteps
    for t=2:n_obs
        # # Resample and rejuvenate if the effective sample size is too low
        if effective_sample_size(state) < 0.5 * n_particles
            # Perform residual resampling, pruning low-weight particles
            pf_resample!(state, :residual)
            # Perform a rejuvenation move on past choices
            pf_rejuvenate!(state, mh, (subtree_proposal, (), subtree_involution))
            pf_rejuvenate!(state, mh, (noise_proposal, ()))
        end
        # Update filter state with new observation at timestep t
        pf_update!(state, (xs[1:t],), (UnknownChange(),), obs_choices[t])
        if mod(t,10) == 0
            println("number of observations: $t")
            callback(state, xs, ys, anim_traj, t)
        end
    end
    return state
end

pf_callback = (state, xs, ys, xs_test, ys_test, anim_traj, t) -> begin
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

# do inference, time it
@time state = particle_filter(xs_train, ys_train, n_particles, pf_callback, anim_traj)

make_animation_sequential(animation_name, anim_traj, xs_train, ys_train)
