# include("sequential.jl")
# dataset_name = "quadratic"
# animation_name = "sequential_" * dataset_name
# n_particles = 100
# animation_fn = make_animation_sequential

include("acquisition.jl")
dataset_name = "cubic"
animation_name = "acquisition_" * dataset_name
n_particles = 50
animation_fn = make_animation_acquisition


function run_inference(dataset_name, animation_name, n_particles, animation_fn)
    # load the dataset
    (xs, ys) = get_dataset(dataset_name)
    # (xs, ys) = get_airline_dataset()
    xs_train = xs[1:100]
    ys_train = ys[1:100]
    xs_test = xs[101:end]
    ys_test = ys[101:end]

    # visualization
    anim_traj = Dict()

    # set seed
    Random.seed!(1)

    # do inference, time it
    @time state = particle_filter(xs_train, ys_train, n_particles, pf_callback, anim_traj)

    make_animation_sequential(animation_name, anim_traj)
end

run_inference(dataset_name, animation_name, n_particles, animation_fn)
