using Plots
gr()
Plots.GRBackend()
include("../utils/shared.jl")
include("../acquisition_exploration_AL.jl")
include("functions.jl")

# Generate data
function get_dataset(f, xs_train)
    # xs_train = collect(LinRange(data_bounds[1], data_bounds[2], n_obs))
    sort!(xs_train)
    ys_train = deepcopy(xs_train)
    @. ys_train = f.(xs_train)
    return (xs_train, ys_train)
end

function calc_info_gain(theta1, theta2, noise, new_x)
    conditional_mu_1, conditional_cov_matrix_1 = compute_predictive(theta1, noise, obs_xs, obs_ys, [new_x])
    conditional_mu_2, conditional_cov_matrix_2 = compute_predictive(theta2, noise, obs_xs, obs_ys, [new_x])

    mu_cov = [(conditional_mu_1, conditional_cov_matrix_1), (conditional_mu_2, conditional_cov_matrix_2)]
    thetas = [theta1, theta2]
    weights = [0.5, 0.5]
    p_theta = 0.5

    info_gain = 0

    for i=1:2
        theta_info_gain = 0
        theta = thetas[i]
        conditional_mu, conditional_cov_matrix = mu_cov[i]
        mu, var = conditional_mu[1], conditional_cov_matrix[1,1]
        m_outcomes = [normal(mu, var) for t=1:m]
        for y in m_outcomes
            log_py_trace = log(calc_log_prob_y_given_th(y, mu_cov, i))
            theta_info_gain += log(p_theta) + log_py_trace - calc_log_prob_y(mu_cov, y, weights)
        end
        info_gain += p_theta * 1/m * theta_info_gain
    end
    return info_gain
end

function plot_info_gain(func_xs, func_ys, obs_xs, obs_ys, info_plot)
    # make double plot
    x_min, x_max = minimum(func_xs), maximum(func_xs)
    y_min, y_max = minimum(func_ys), maximum(func_ys)
    l = @layout [a; b]

    # plot function
    p = plot(func_xs, func_ys, title="Info Gain Test Case ", xlim=(x_min, x_max), ylim=(y_min-1, y_max+1), legend=false, linecolor=:red, layout = l)
    # plot observations
    old_obs = length(obs_xs) - 1
    plot!(p[1], obs_xs, obs_ys, seriestype = :scatter,  marker = (:circle, 0.4, 8, :yellow))

    # plot info gain
    if length(info_plot) > 0
        max_info_idx = argmax(info_plot)
        low_bound = minimum(info_plot) - 1
        plot!(p[2], func_xs, info_plot, fillrange=[[low_bound for i=1:length(info_plot)], info_plot], fillalpha=0.6, fillcolor=:green, title="Information Gain", xlim=(x_min, x_max), ylim=(low_bound, maximum(info_plot) + 1), legend=false)
        plot!(p[2], [func_xs[max_info_idx]], [info_plot[max_info_idx]],  seriestype = :scatter,  marker = (:star, 0.8, 8, :red))
    end

    savefig(p, "plots_gp/test_infogain_plot")
end

# obs_xs, obs_ys = [0.0, 2.0], [1.0, 1.0]
obs_xs, obs_ys = get_dataset(functions["quadratic"], [0.0, 2.0])
m = 500

quadratic_fn = Times(Linear(1.0), Linear(1.0))
constant_fn = Constant(2.0)
noise = 0.0001
pred_xs = collect(LinRange(-3.0, 5.0, 100))
sort!(pred_xs)

func_xs, func_ys = get_dataset(functions["quadratic"], pred_xs)
info_plot = [calc_info_gain(quadratic_fn, constant_fn, noise, new_x) for new_x in pred_xs]

plot_info_gain(func_xs, func_ys, obs_xs, obs_ys, info_plot)
