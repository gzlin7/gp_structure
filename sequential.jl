include("utils/shared.jl")
include("utils/unfold_model.jl")
include("utils/plotting.jl")
using GenParticleFilters
@load_generated_functions()

function particle_filter_sequential(xs::Vector{Float64}, ys::Vector{Float64}, n_particles, callback, anim_traj)
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
