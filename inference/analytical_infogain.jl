using Plots
using Dstributions
gr()
Plots.GRBackend()
include("shared.jl")
include("acquisition_exploration_AL.jl")

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
    p_theta = 0.5
    for theta in [theta1, theta2]
        theta_info_gain = 0
        conditional_mu, conditional_cov_matrix = compute_predictive(node, noise, obs_xs, obs_ys, [new_x])
        mu, var = conditional_mu[1], conditional_cov_matrix[1,1]
        m_outcomes = [normal(mu, var) for t=1:m]
        conditional_dist = MvNormal(conditional_mu, conditional_cov_matrix)
        for y in m_outcomes
            log_py_trace = log(calc_prob_y_given_th(y, [conditional_dist], 1))
            theta_info_gain += log(p_theta) + log_py_trace - log(calc_prob_y(mvnorms, y, weights))
        end
        info_gain += p_theta * 1/2 * theta_info_gain
    end
    return info_gain
end

plot_name = "analytical_infogain"
quadratic_fn = Times(Linear(1.0), Linear(1.0))
constant_fn = Constant(2.0)
noise = 0.0001
obs1, obs2 = (0.0, 1.0), (2.0, 1.0)
pred_xs = collect(LinRange(-1.0, 3.0, 100))
sort!(pred_xs)

print([calc_info_gain(quadratic_fn, constant_fn, noise, new_x) for new_x in pred_xs])

# test_plot_gp(plot_name, quadratic_fn, constant_fn, pred_xs, noise)
# test_plot_gp_fixed(plot_name, quadratic_fn, constant_fn, [0.0, 2.0], [1.0, 1.0], pred_xs, noise)
