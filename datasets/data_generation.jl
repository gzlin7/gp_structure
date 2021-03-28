using Gen
using DataFrames
using CSV

# data generating functions
function quadratic(x)
    return (x - 0.5) ^ 2 + normal(noise_mu, noise_std)
end
function cubic(x)
    # 20 * (x - 0.2)(x - 0.6)(x - 1) + 0.5
    return 20 * ((x - 0.2)*(x - 0.6)*(x - 1)) + 0.5 + normal(noise_mu, noise_std)
end

filename = "cubic"
f = cubic
n_train = 100
n_test = 100
noise_mu = 0
noise_std = 0.01

# training data
train_xs = [uniform(0.0,1.0) for t=1:n_train]
train_ys = deepcopy(train_xs)
@. train_ys = f.(train_ys)
train_df = DataFrame(X=train_xs, Y=train_ys)
sort!(train_df)

# test data
test_xs = [uniform(0.0,1.0) for t=1:n_test]
test_ys = deepcopy(test_xs)
@. test_ys = f.(test_ys)
test_df = DataFrame(X=test_xs, Y=test_ys)
sort!(test_df)

combined_df = vcat(train_df, test_df)

CSV.write("C:\\classes\\gp_structure\\" * filename * ".csv", combined_df)
