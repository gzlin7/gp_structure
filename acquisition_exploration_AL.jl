include("utils/shared.jl")
include("utils/unfold_model.jl")
include("utils/plotting.jl")
include("AL_objectives.jl")
using GenParticleFilters
using Optim
using StatsBase
@load_generated_functions()

# objectives defined in AL_objectives.jl
objective_name_to_func = Dict("random" => random_obj, "sequential" => sequential_obj, "info_gain" => gp_obj, "max_variance" => max_variance, "region_of_interest" => region_of_interest)
function particle_filter_acquisition_AL(xs::Vector{Float64}, ys::Vector{Float64}, n_particles, callback, anim_traj, x_obs_traj, y_obs_traj, budget, objective::String="random")
    # keep track of xs we have not evaluated yet
    potential_xs = deepcopy(xs)
    n_explore = 0
    obs_idx = []
    obs_xs = Float64[]
    obs_ys = Float64[]

    obs_choices = [choicemap()]
    state = pf_initialize(model, (1,), obs_choices[1], n_particles)

    get_next_obs_x = objective_name_to_func[objective]

    if objective == "sequential"
        step = Int(floor(length(potential_xs)/budget))
        sort!(potential_xs)
        potential_xs = potential_xs[1:step:end]
    end

    # Iterate across timesteps
    for t=2:budget
        # Resample and rejuvenate if the effective sample size is too low
        # if effective_sample_size(state) < 0.5 * n_particles
        # Perform residual resampling, pruning low-weight particles
        if (mod(t,5) == 0)
            pf_resample!(state, :multinomial)
            # Perform a rejuvenation move on past choices
            pf_rejuvenate!(state, mh, (subtree_proposal, (), subtree_involution))
            pf_rejuvenate!(state, mh, (noise_proposal, ()))
        end

        # select next observation point
        if (t > n_explore)
            ret = get_next_obs_x(state, potential_xs, obs_xs, obs_ys)
            potential_xs_idx, xs_info_plot, info_plot = ret
        end

        # convwert from index in remaining x's to index in xs array
        next_x = potential_xs[potential_xs_idx]
        deleteat!(potential_xs, potential_xs_idx)
        next_obs_idx = findfirst(isequal(next_x), xs)

        # keep track of observations for animation
        push!(obs_idx, next_obs_idx), push!(obs_xs, xs[next_obs_idx]), push!(obs_ys, ys[next_obs_idx])
        push!(x_obs_traj, xs[next_obs_idx]), push!(y_obs_traj, ys[next_obs_idx])

        # Update filter state with new observation at timestep t
        push!(obs_choices, choicemap((:state => t => :x, xs[next_obs_idx]), (:state => t => :y, ys[next_obs_idx])))
        pf_update!(state, (t,), (UnknownChange(),), obs_choices[t])

        if mod(t,5) == 0
            # println(obs_idx)
            println("number of observations: $t")
            # print top cov kernels
            ret = sample_traces(state, 100)
            traces = ret[1]
            weights = ret[2]
            best_idxes = get_n_top_weight_idxs(1, weights)
            for idx in best_idxes
                println(get_retval(traces[idx])[1])
            end
        end

        callback(state, xs, ys, anim_traj, t, xs_info_plot, info_plot)
    end
    return state
end
