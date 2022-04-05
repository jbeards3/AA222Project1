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
using LinearAlgebra

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
        x_history, f_history = optimizer_simple(f, g, x0, n, 0.24)
        #x_history, f_history = optimizer_momentum(f, g, x0, n, 1, 0.2)
    elseif prob == "simple2"
        x_history, f_history = optimizer_simple(f, g, x0, n, 0.3)
    else
        x_history, f_history = optimizer_simple(f, g, x0, n, 0.1)
    end
    x_best = x_history[argmin(f_history)]
    return x_best
end

function optimizer_simple(f, g, x0, n, step = 0.01)
    x_history = [x0]
    f_history = [f(x0)]

    while count(f, g) < n - 1
        # Calculate local descent
        g_x = g(x_history[end])
        d = -g_x / norm(g_x) 
        # step = argmin(f(x_history[end] + step * d))

        # Calculate next design point
        x_next = x_history[end] + step * d
        # x_next = x_history[end] - step * g_x

        # Add design point to history of design points
        push!(x_history, x_next)
        push!(f_history, f(x_next))
    end

    return x_history, f_history
end

function optimizer_momentum(f, g, x0, n, step = 1, Beta = 0)
    x_history = [x0]
    f_history = [f(x0)]
    v = zeros(length(x0))

    for i = 1:3:n-3
        # Calculate local descent
        g_x = g(x_history[end])
        d = -g_x / norm(g_x)
        v[:] = Beta*v + step * d

        # Calculate next design point
        x_next = x_history[end] + v

        # Add design point to history of design points
        push!(x_history, x_next)
        push!(f_history, f(x_next))
    end

    return x_history, f_history
end
