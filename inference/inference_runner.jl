using Random
using HDF5, JLD
include("inference_helper.jl")

# dataset_names = [ "02", "04", "05", "10", "21"]
dataset_names = [ "10"]
# dataset_names = ["periodic"]
# dataset_names = ["sq_exp"]
# dataset_names = ["21", "02"]
objective_functions = ["max_variance", "region_of_interest", "sequential", "random", "info_gain"]
# objective_functions = ["max_variance", "sequential"]
obj_to_info_metric = Dict("info_gain" => "Information Gain", "max_variance" => "Total Predictive Variance", "region_of_interest" => "Mean Marginal Entropy")
n_particles_all = [100]
n_trials = 1

function makepath(rel_path)
    if isdir(rel_path) == false
        mkpath(rel_path)
    end
end

# save run log
function save_run_log(dataset_name, inference_ret_all)
    run_log_name = dataset_name * "_" * randstring(10)
    # open("animations/acquisition/" * run_log_name * ".txt", "w") do file
    #     println(file, string(inference_ret_all))
    # end
    save("animations/acquisition/" * run_log_name * ".jld", inference_ret_all)
end

for n_particles in n_particles_all
    for i=1:length(dataset_names)
        dataset_name = dataset_names[i]
        n_obs_plotting = haskey(fn_to_obs, dataset_name) ? fn_to_obs[dataset_name] : n_obs_default
        budget = 20
        inference_ret_all = Dict()

        for objective in objective_functions
            inference_ret_all[objective] = []
        end

        noise = 0.0001
        if dataset_name == "periodic"
            cov_fn = Periodic(0.25, 0.5)
            pred_xs = sort!(collect(LinRange(-1, 1, 200)))
            train_x, train_y = get_dataset_gp(cov_fn, noise, pred_xs)
            data = [train_x, train_y]
        elseif dataset_name == "sq_exp"
            cov_fn = SquaredExponential(0.1)
            pred_xs = sort!(collect(LinRange(-10, 10, 200)))
            train_x, train_y = get_dataset_gp(cov_fn, noise, pred_xs)
            data = [train_x, train_y]
        else
            data = load_data(dataset_name, n_obs_plotting)
        end

        rescale data y to -1, 1
        dataset_max, dataset_min = maximum(data[2]), minimum(data[2])
        range = dataset_max - dataset_min

        if range > 2
            scaled_ys = data[2] .* (2 / range)
            # shift to x axis
            if maximum(scaled_ys) > 1
                scaled_ys - scaled_ys .- (maximum(scaled_ys) - 1)
            elseif minimum(scaled_ys) < -1
                scaled_ys - scaled_ys .- (minimum(scaled_ys) + 1)
            end
            data = (data[1], scaled_ys)
        end

        for j=1:n_trials
            for objective in objective_functions
                println(" ")
                println("objective " * objective)
                # do inference
                inference_ret =  run_inference(data, n_particles, budget, objective)
                info_metric = haskey(obj_to_info_metric, objective) ? obj_to_info_metric[objective] : " "
                # visualize_inference(animation_name, objective, info_metric, inference_ret)
                # save animation trajectory for overal plot
                push!(inference_ret_all[objective], inference_ret)
            end
        end
        save_run_log(dataset_name, inference_ret_all)
        rel_path = "animations/acquisition/" * dataset_name * "/run"
        makepath(rel_path)
        make_animation_acquisition_multi(rel_path, obj_to_info_metric, inference_ret_all)
        # plot combined graph
        plot_name = dataset_name * "_" * string(n_trials) * "_" * string(n_particles)
        make_acc_plot_multi(plot_name, inference_ret_all)
    end
end
