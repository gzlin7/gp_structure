include("inference_helper.jl")

# plot 1 function
# dataset_name, animation_name, n_particles, n_obs_plotting, budget = "21", "quadratic_anim", 100, 100, 20
# #load data
# data = load_data(dataset_name, n_obs_plotting)
# # run inference
# inference_ret = run_inference(data, n_particles, budget, false)
# # visualize
# visualize_inference(animation_name, inference_ret)

# dataset_names = ["changepoint", "polynomial", "sinusoid", "quadratic", "linear","airline", "quadratic"]
dataset_names = ["21", "02", "05", "10", "04"]
dataset_names = ["21"]
# dataset_names = ["quadratic"]
n_particles_all = [100]
n_trials = 1

for i=1:length(dataset_names)
    dataset_name = dataset_names[i]
    n_obs_plotting = haskey(fn_to_obs, dataset_name) ? fn_to_obs[dataset_name] : n_obs_default
    budget = 10

    anim_traj_random = []
    anim_traj_AL = []

    for j=1:n_trials

        char = "multi_"

        animation_name_rand = char * dataset_name * "_rand_" * string(n_particles) * "_" * string(j)
        animation_name_al = char * dataset_name * "_active_" * string(n_particles) * "_" * string(j)

        data = load_data(dataset_name, n_obs_plotting)

        # random
        rand_inference_ret =  run_inference(data, n_particles, budget, true)
        push!(anim_traj_random, rand_inference_ret[1])
        visualize_inference(animation_name_rand, rand_inference_ret)
        # AL
        al_inference_ret =  run_inference(data, n_particles, budget, false)
        push!(anim_traj_AL, al_inference_ret[1])
        visualize_inference(animation_name_al, al_inference_ret)
    end

    # plot combined graph
    plot_name = dataset_name * "_multi_overall_" * string(n_trials) * "_" * string(n_particles)
    make_acc_plot_multi(plot_name, anim_traj_random, anim_traj_AL)
end
