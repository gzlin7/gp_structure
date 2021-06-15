using Plots
using Dstributions
gr()
Plots.GRBackend()
include("shared.jl")

# Generate data
function get_dataset(f, n_obs, data_bounds)
    xs_train = collect(LinRange(data_bounds[1], data_bounds[2], n_obs))
    sort!(xs_train)
    ys_train = deepcopy(xs_train)
    @. ys_train = f.(xs_train)
    return (xs_train, ys_train)
end

obs_xs, obs_ys = [0.0, 2.0], [1.0, 1.0]

function calc_info_gain(theta1, theta2, noise, new_x)
    info_gain = 0
    for theta in [theta1, theta2]
        conditional_mu, conditional_cov_matrix = compute_predictive(node, noise, obs_xs, obs_ys, [new_x])
        conditional_dist = MvNormal(conditional_mu, conditional_cov_matrix)
        for


    end
end


plot_name = "analytical_infogain"
quadratic_fn = Times(Linear(1.0), Linear(1.0))
constant_fn = Constant(2.0)
noise = 0.0001
obs1, obs2 = (0.0, 1.0), (2.0, 1.0)
pred_xs = collect(LinRange(-1.0, 3.0, 100))
sort!(pred_xs)

# test_plot_gp(plot_name, quadratic_fn, constant_fn, pred_xs, noise)
# test_plot_gp_fixed(plot_name, quadratic_fn, constant_fn, [0.0, 2.0], [1.0, 1.0], pred_xs, noise)
