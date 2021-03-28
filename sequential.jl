include("utils/shared.jl")
include("utils/unfold_model.jl")
include("utils/plotting.jl")
using GenParticleFilters
using Plots
gr()
Plots.GRBackend()
@load_generated_functions()

dataset_name = "cubic"
animation_name = "sequential_" * dataset_name
n_particles = 100

function particle_filter(xs::Vector{Float64}, ys::Vector{Float64}, n_particles, callback, anim_traj)
    n_obs = length(xs)
    obs_choices = [choicemap((:state => t => :x, xs[t]), (:state => t => :y, ys[t])) for t=1:n_obs]
    state = pf_initialize(model, (1,), obs_choices[1], n_particles)
    # Iterate across timesteps
    for t=2:n_obs
        # Resample and rejuvenate if the effective sample size is too low
        if effective_sample_size(state) < 0.5 * n_particles
            # Perform residual resampling, pruning low-weight particles
            pf_resample!(state, :residual)
            # Perform a rejuvenation move on past choices
            pf_rejuvenate!(state, mh, (subtree_proposal, (), subtree_involution))
            pf_rejuvenate!(state, mh, (noise_proposal, ()))
        end
        # Update filter state with new observation at timestep t
        pf_update!(state,(t,), (UnknownChange(),), obs_choices[t])
        if mod(t,10) == 0
            println("number of observations: $t")
            callback(state, xs, ys, anim_traj, t)
        end
    end
    return state
end

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

# load the dataset
(xs, ys) = get_dataset(dataset_name)
# (xs, ys) = get_airline_dataset()
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

make_animation_sequential(animation_name, anim_traj)
