functions = Dict("sinusoid"=> x -> 0.20sin(15.7x),
                 "linear"=> x -> 0.4x,
                 "quadratic" => x -> (x-1)^2,
                 "polynomial" => x -> -0.2 * (x - 0.3)^2 * (x - 3.3) * (x - 4.2) * (x - 1.3) * (x - 3.7) * (x + 0.4) * (x - 2.1),
                 "cubic" => x -> 20 * ((x - 0.2)*(x - 0.6)*(x - 1)) + 0.5,
                 "changepoint" => x -> x < 2.0 ? 4(x - 1) ^ 2 : 0.20sin(15.7x),
                 "airline" => get_airline_dataset,
                 # http://infinity77.net/global_optimization/test_functions_1d.html
                 "02" => x -> sin(x) + sin(10/3 * x),
                 "03" => x -> -sum([k * sin((k + 1) * x + k) for k=1:6]),
                 "04" => x -> -(16*x^2 - 24*x + 5) * exp(-x),
                 "05" => x -> -(1.4-3*x)*sin(18*x),
                 "06" => x -> -(x + sin(x)) * exp(-x^2),
                 "07" => x -> sin(x) + sin(10/3 * x) + log(x) - 0.84*x + 3,
                 "08" => x -> -sum([k * cos((k + 1) * x + k) for k=1:6]) ,
                 "10" => x -> -x * sin(x),
                 "14" => x -> -exp(-x) * sin(2*pi*x),
                 "21" => x -> x * sin(x) + x * cos(2x)             )

bounds_default = (0.0,0.4)
bounds =  Dict( "quadratic" => (-1.0, 3.0),
                "02" =>  (2.7,7.5),
                "03" => (-10, 10),
                "04" => (1.9, 3.9),
                "05" => (0, 1.2),
                "06" => (-10, 10),
                "07" => (2.7, 7.5),
                "08" => (-10, 10),
                "09" => (3.1, 20.4),
                "10" => (0,10),
                "14" => (0,4),
                "21" => (0,10)
                 )

n_obs_default = 100
fn_to_obs = Dict("linear"=> 50,
                 "03" => 500,
                 "06" => 500
                  )
