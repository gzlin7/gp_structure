using Gen
using DataFrames
using CSV

n_obs = 100
noise_mu = 0
noise_var = 1

# data generating function
function f(x)
    return (x - 0.5) ^ 2 +
end


xs = [uniform(0.0,1.0) for t=1:n_obs]
ys = deepcopy(xs)



@. ys = f.(ys)

df = DataFrame(X=xs, Y=ys)
sort!(df)

CSV.write("C:\\classes\\gp_structure\\test_data.csv", df)
