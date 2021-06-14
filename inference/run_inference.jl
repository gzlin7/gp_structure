include("inference_helper.jl")

# plot 1 function
dataset_name, animation_name, n_particles, n_obs_plotting, budget = "quadratic", "quadratic_anim", 100, 100, 20
#load data
data = load_data(dataset_name, n_obs_plotting)
# run inference
inference_ret = run_inference(data, n_particles, budget, false)
# visualize
visualize_inference(animation_name, inference_ret)

# dataset_names = ["changepoint", "polynomial", "sinusoid", "quadratic", "linear","airline", "quadratic"]
# dataset_names = ["21", "02", "05", "10", "04"]
# dataset_names = ["quadratic"]
# n_particles_all = [100]
#
#
# for i=1:length(dataset_names)
#     dataset_name = dataset_names[i]
#     n_obs_plotting = haskey(fn_to_obs, dataset_name) ? fn_to_obs[dataset_name] : n_obs_default
#     budget = 10
#
#     # # run sequential prediction
#     for n_particles in n_particles_all
#
#         # run acquisition prediction
#         sequential = false
#
#         char = "m"
#
#         animation_name_rand = char * dataset_name * "_rand_" * string(n_particles)
#         animation_name_al = char * dataset_name * "_active_" * string(n_particles)
#
#         # run_inference(dataset_name, animation_name_rand, n_particles, sequential, functions[dataset_name], n_obs_plotting, budget, true)
#         # run_inference(dataset_name, animation_name_al, n_particles, sequential, functions[dataset_name], n_obs_plotting, budget, false)
#     end
# end
