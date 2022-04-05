#=
        project1.jl -- This is where the magic happens!

    All of your code must either live in this file, or be `include`d here.
=#

#=
    If you want to use packages, please do so up here.
    Note that you may use any packages in the julia standard library
    (i.e. ones that ship with the julia language) as well as Statistics
    (since we use it in the backend already anyway)
=#

# Example:
# using LinearAlgebra

#=
    If you're going to include files, please do so up here. Note that they
    must be saved in project1_jl and you must use the relative path
    (not the absolute path) of the file in the include statement.

    [Good]  include("somefile.jl")
    [Bad]   include("/pathto/project1_jl/somefile.jl")
=#

# Example
# include("myfile.jl")


"""
    optimize(f, g, x0, n, prob)

Arguments:
    - `f`: Function to be optimized
    - `g`: Gradient function for `f`
    - `x0`: (Vector) Initial position to start from
    - `n`: (Int) Number of evaluations allowed. Remember `g` costs twice of `f`
    - `prob`: (String) Name of the problem. So you can use a different strategy for each problem. E.g. "simple1", "secret2", etc.

Returns:
    - The location of the minimum
"""
function optimize(f, g, x0, n, prob)
    if prob == "simple1"
        x_history, f_history = optimizer(f, g, x0, n)
    elseif prob == "simple2"
        x_history, f_history = optimizer(f, g, x0, n)
    else
        x_history, f_history = optimizer(f, g, x0, n)
    end
    x_best = x_history[argmin(f_history)]
    return x_best
end

function optimizer(f, g, x0, n, step = 0.1)
    x_history = [x0]
    f_history = [f(x0)]

    for i = 1:n-1
        x_next = x_history[end] + step * ones(length(x0))
        push!(x_history, x_next)
        push!(f_history, f(x_next))
    end

    return x_history, f_history
end

