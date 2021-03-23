using Gen
using DataFrames
using CSV

filename = "test_data"
n_train = 100
n_test = 100
noise_mu = 0
noise_std = 0.01
# data generating function
function f(x)
    return (x - 0.5) ^ 2 + normal(noise_mu, noise_std)
end

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
