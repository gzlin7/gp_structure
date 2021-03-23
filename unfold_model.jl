using Gen
import LinearAlgebra
import CSV
import Random
using GenParticleFilters


@gen function covariance_prior(cur::Int)
    node_type = @trace(categorical(node_dist), (cur, :type))

    if node_type == CONSTANT
        param = @trace(uniform_continuous(0, 1), (cur, :param))
        node = Constant(param)

    # linear kernel
    elseif node_type == LINEAR
        param = @trace(uniform_continuous(0, 1), (cur, :param))
        node = Linear(param)

    # squared exponential kernel
    elseif node_type == SQUARED_EXP
        length_scale= @trace(uniform_continuous(0, 1), (cur, :length_scale))
        node = SquaredExponential(length_scale)

    # periodic kernel
    elseif node_type == PERIODIC
        scale = @trace(uniform_continuous(0, 1), (cur, :scale))
        period = @trace(uniform_continuous(0, 1), (cur, :period))
        node = Periodic(scale, period)

    # plus combinator
    elseif node_type == PLUS
        child1 = get_child(cur, 1, max_branch)
        child2 = get_child(cur, 2, max_branch)
        left = @trace(covariance_prior(child1))
        right = @trace(covariance_prior(child2))
        node = Plus(left, right)

    # times combinator
    elseif node_type == TIMES
        child1 = get_child(cur, 1, max_branch)
        child2 = get_child(cur, 2, max_branch)
        left = @trace(covariance_prior(child1))
        right = @trace(covariance_prior(child2))
        node = Times(left, right)

    # unknown node type
    else
        error("Unknown node type: $node_type")
    end

    return node
end

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
    push!(xs, x)
    var = eval_cov(covariance_fn, x, x) + noise

    i = t
    if i == 1
        mu = 0.0
        covm_22_inv = reshape([1/var], 1, 1)
        cov_matrix = reshape([var], 1, 1)
    end
    if i > 1
        xs = xs[1:i]
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
    state ~ get_conditional_ys(n, [sampled_xs,ys,mus,vars,covm_22, covariance_matrix], covariance_fn, noise)
    ys = last(state)[2]
    return (covariance_fn, ys)
end

@gen function noise_proposal(prev_trace)
    @trace(gamma(1, 1), :noise)
end

@gen function subtree_proposal(prev_trace)
    prev_subtree_node::Node = get_retval(prev_trace)[1]
    (subtree_idx::Int, depth::Int) = @trace(pick_random_node(prev_subtree_node, 1, 0), :choose_subtree_root)
    new_subtree_node::Node = @trace(covariance_prior(subtree_idx), :subtree)
    (subtree_idx, depth, new_subtree_node)
end

function subtree_involution(trace, fwd_choices::ChoiceMap, fwd_ret::Tuple, proposal_args::Tuple)
    (subtree_idx, subtree_depth, new_subtree_node) = fwd_ret
    model_args = get_args(trace)

    # populate constraints with proposed subtree
    constraints = choicemap()
    set_submap!(constraints, :tree, get_submap(fwd_choices, :subtree))

    # populate backward assignment with choice of root
    bwd_choices = choicemap()
    set_submap!(bwd_choices, :choose_subtree_root => :recurse_left,
        get_submap(fwd_choices, :choose_subtree_root => :recurse_left))
    for depth=0:subtree_depth-1
        bwd_choices[:choose_subtree_root => :done => depth] = false
    end
    if !isa(new_subtree_node, LeafNode)
        bwd_choices[:choose_subtree_root => :done => subtree_depth] = true
    end

    # obtain new trace and discard, which contains the previous subtree
    (new_trace, weight, _, discard) = update(trace, model_args, (NoChange(),), constraints)

    # populate backward assignment with the previous subtree
    set_submap!(bwd_choices, :subtree, get_submap(discard, :tree))

    (new_trace, bwd_choices, weight)
end
