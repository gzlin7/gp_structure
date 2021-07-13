include("../utils/plotting.jl")
using JLD, HDF5

# run_log_names = ["02_ShD9rOZgtG", "04_D4R2jA5Dep", "05_eO7CgHMAWE", "05_eO7CgHMAWE", "21_qCMXy9XOLx"]
# run_log_names = ["sq_exp_SndEulUaac"]
# run_log_names = ["periodic_VEiv1zhhcE"]
run_log_names = ["10_wLUQvv3i2b"]
obj_to_info_metric = Dict("info_gain" => "Information Gain", "max_variance" => "Total Predictive Variance", "region_of_interest" => "Mean Marginal Entropy")

visualization_path = "animations/acquisition/"

# open("animations/acquisition/" * run_log_name * ".txt", "r") do file
#     parsed_str = read(file, String)
#     global inference_ret_all = eval(parse(parsed_str))
# end

for run_log_name in run_log_names
    inference_ret_all = load(visualization_path * run_log_name * ".jld")
    dataset = run_log_name[1:2]
    make_animation_acquisition_multi("animations/acquisition/" * dataset * "/run", obj_to_info_metric, inference_ret_all)
end

# inference_ret_all = load(visualization_path * "sq_exp_SndEulUaac" * ".jld")
# make_acc_plot_multi("sq_exp_10_100", inference_ret_all)
