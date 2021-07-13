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

# plot posterior means and vanriance for each cov fn
function test_plot_gp(plot_name, cov_fns, xs, noise)
    p = plot(legend=true, title=plot_name)
    for cov_fn in cov_fns
        ys = mvnormal(zeros(length(xs)), compute_cov_matrix(cov_fn, noise, xs))
        plot!(p, xs, ys, label=string(cov_fn))
    end
    savefig(p, "plots_gp/" * plot_name)
end

function test_plot_gp_w_obs(plot_name, fn_xs, fn_ys, cov_fns, past_xs, past_ys, xs, noise)
    # plot posterior means and vanriance given one covariance fn
    p = plot(legend=true, title=plot_name)
    for cov_fn in cov_fns
        ys = predict_ys(cov_fn, noise, past_xs, past_ys, xs)
        plot!(p, xs, ys, label=string(cov_fn))
    end
    # plot function
    plot!(p, fn_xs, fn_ys, label="test_fn")
    savefig(p, "plots_gp/" * plot_name)
end

noise = 0.0001

# test case 1: quadratic vs constant
plot_name = "test_1"
cov_fns = [Times(Linear(1.0), Linear(1.0)),  Constant(2.0)]
pred_xs = sort!(collect(LinRange(-1.0, 3.0, 150)))
f = x -> (x-1)^2
fn_xs, fn_ys = get_dataset(f, 100, [-1, 3])
test_plot_gp_w_obs(plot_name, fn_xs, fn_ys, cov_fns, [0.0, 2.0], [1.0, 1.0], pred_xs, noise)

# test case 2: fixed SE/RBF kernel w lengthscale 0.1
plot_name = "test_2"
lengthscales = [0.1, 0.5, 1.0]
cov_fns = [SquaredExponential(l) for l in lengthscales]
pred_xs = sort!(collect(LinRange(-10, 10, 200)))
# test_plot_gp(plot_name, cov_fns, pred_xs, noise)

# test case 3: fixed Periodic kernel w fixed amplitude
plot_name = "test_3"
periods = [0.1, 0.5, 1.0]
scale = 0.5
cov_fns = [Periodic(scale, period) for period in periods]
pred_xs = sort!(collect(LinRange(-10, 10, 200)))
test_plot_gp(plot_name, cov_fns, pred_xs, noise)
