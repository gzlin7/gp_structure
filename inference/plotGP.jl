using Plots
gr()
Plots.GRBackend()
include("../utils/shared.jl")

# Generate data
function get_dataset(f, n_obs, data_bounds)
    xs_train = collect(LinRange(data_bounds[1], data_bounds[2], n_obs))
    sort!(xs_train)
    ys_train = deepcopy(xs_train)
    @. ys_train = f.(xs_train)
    return (xs_train, ys_train)
end

function test_plot_gp(plot_name, covariance_fn_1, covariance_fn_2, xs, noise)
    # plot posterior means and vanriance given one covariance fn
    ys_1 = mvnormal(zeros(length(xs)), compute_cov_matrix(covariance_fn_1, noise, xs))
    ys_2 = mvnormal(zeros(length(xs)), compute_cov_matrix(covariance_fn_2, noise, xs))

    p = plot(xs, ys_1, linecolor=:teal, fillalpha=0.5, fillcolor=:lightblue)
    plot!(p, xs, ys_2, linecolor=:red, fillalpha=0.5, fillcolor=:lightblue)

    savefig(p, "plots_gp/" * plot_name)
end

function test_plot_gp_fixed(plot_name, covariance_fn_1, covariance_fn_2, past_xs, past_ys, xs, noise)
    # plot posterior means and vanriance given one covariance fn
    ys_1 =   predict_ys(covariance_fn_1, noise, past_xs, past_ys, xs)
    ys_2 = predict_ys(covariance_fn_2, noise, past_xs, past_ys, xs)

    p = plot(xs, ys_1, linecolor=:teal, fillalpha=0.5, fillcolor=:lightblue)
    plot!(p, xs, ys_2, linecolor=:red, fillalpha=0.5, fillcolor=:lightblue)

    fn_xs, fn_ys = get_dataset(x -> (x-1)^2, 100, [-1, 3])
    plot!(p, fn_xs, fn_ys, linecolor=:green, fillalpha=0.5, fillcolor=:lightblue)

    savefig(p, "plots_gp/" * plot_name)
end

plot_name = "testing_gp_plot"
quadratic_fn = Times(Linear(1.0), Linear(1.0))
constant_fn = Constant(2.0)
noise = 0.0001

pred_xs = collect(LinRange(-1.0, 3.0, 150))
sort!(pred_xs)

# test_plot_gp(plot_name, quadratic_fn, constant_fn, pred_xs, noise)
test_plot_gp_fixed(plot_name, quadratic_fn, constant_fn, [0.0, 2.0], [1.0, 1.0], pred_xs, noise)
