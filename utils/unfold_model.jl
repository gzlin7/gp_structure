using Gen
import LinearAlgebra
import CSV
import Random
using GenParticleFilters
include("subtree.jl")


function blockwise_inv(cov_matrix, prev_inv_22, i)
    a_inv = prev_inv_22
    b = reshape(cov_matrix[i-1,1:i-2],i-2,1)
    c = transpose(b)
    d = reshape([cov_matrix[i-1, i-1]],1,1)
    covm_22_inv = hcat(a_inv + a_inv*b*inv(d-(c*a_inv*b))*c*a_inv, -a_inv*b*inv(d-c*a_inv*b))
    covm_22_inv = vcat(covm_22_inv, hcat(-inv(d-c*a_inv*b)*c*a_inv, inv(d-c*a_inv*b)))
    return covm_22_inv
end

function cov_matrix_incremental(old_cov_matrix, covariance_fn, xs, var)
    n = size(old_cov_matrix)[1]
    new_x = last(xs)
    # calculate new covariances
    cov_matrix = zeros(Float64, n+1, n+1)
    cov_matrix[1:n, 1:n] = old_cov_matrix
    new_col = float(range(1,n,step=1) |> collect)
    function f(idx)
        eval_cov(covariance_fn, xs[idx], new_x)
    end
    @. new_col = f.(Int(new_col))
    # new_col = zeros(Float64, n, 1)
    # for i=1:n
    #     new_col[i, 1] = eval_cov(covariance_fn, xs[i], new_x)
    # end
    cov_matrix[n+1, 1:n] = transpose(new_col)
    cov_matrix[1:n, n+1] = new_col
    cov_matrix[n+1,n+1] = var
    return cov_matrix
end

@gen function calc_conditional_dist(t, prev_state, covariance_fn, noise)
    # state- xs, ys, mus, vars, covm_22_inv, covm_full
    xs = prev_state[1]
    ys = prev_state[2]
    mus = prev_state[3]
    vars = prev_state[4]
    covm_22_inv = prev_state[5]
    cov_matrix = prev_state[6]

    # randomly sample x
    x ~ uniform(0,1000)
    xs = [xs; x]
    var = eval_cov(covariance_fn, x, x) + noise

    i = t
    if i == 1
        mu = 0.0
        covm_22_inv = reshape([1/var], 1, 1)
        cov_matrix = reshape([var], 1, 1)
    end
    if i > 1
        # xs = xs[1:i]
        cov_matrix = cov_matrix_incremental(cov_matrix, covariance_fn, xs, var)
        covm_11 = cov_matrix[i,i]
        covm_21 = cov_matrix[1:i-1,i]
        covm_12 = transpose(covm_21)
        if i > 2
            covm_22_inv = blockwise_inv(cov_matrix, covm_22_inv, i)
        end
        mu = (covm_12 * covm_22_inv * ys)[1]
        var = covm_11[1] - (covm_12 * covm_22_inv * covm_21)[1]
    end

    # sample y from conditional distribution
    y ~ normal(mu, sqrt(var))

    ys = [ys; y]
    mus = [mus; mu]
    vars = [vars; var]
    state = [xs, ys, mus, vars, covm_22_inv, cov_matrix]
    return state
end

get_conditional_ys = Unfold(calc_conditional_dist)

@gen (static) function model(n)
    # sample covariance function
    covariance_fn::Node = @trace(covariance_prior(1), :tree)

    # sample diagonal noise
    noise = @trace(gamma(1, 1), :noise) + 0.01

    # sample from multivariate normal
    sampled_xs = Float64[]
    ys = Float64[]
    mus = Float64[]
    vars = Float64[]
    covm_22 = []
    covariance_matrix = []
    vars = Float64[]
    state ~ get_conditional_ys(n, [sampled_xs, ys, mus, vars, covm_22, covariance_matrix], covariance_fn, noise)
    ys = last(state)[2]
    return (covariance_fn, ys)
end

function pf_callback(state, xs_train, ys_train, anim_traj, t)
    # calculate E[MSE]
    n_particles = length(state.traces)
    e_mse = 0
    e_pred_ll = 0
    weights = get_norm_weights(state)
    if haskey(anim_traj, t) == false
        push!(anim_traj, t => [])
    end
    for i=1:n_particles
        trace = state.traces[i]
        covariance_fn = get_retval(trace)[1]
        noise = trace[:noise]
        push!(anim_traj[t], [covariance_fn, noise, weights[i]])
        mse =  compute_mse(covariance_fn, noise, xs_train, ys_train, xs_test, ys_test)
        pred_ll = predictive_ll(covariance_fn, noise, xs_train, ys_train, xs_test, ys_test)
        e_mse += mse * weights[i]
        e_pred_ll += pred_ll * weights[i]
    end
    println("E[mse]: $e_mse, E[predictive log likelihood]: $e_pred_ll")
end
