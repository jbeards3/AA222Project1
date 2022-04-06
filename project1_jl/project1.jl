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

# Will only be logging history of x throughout each iteration since gradient descent method doesn't need f(x),
# plus keeping history of f(x) at each iteration wastes an evaluation, thus reducing the number of iterations
# I can run. Since the number of evaluation doesn't matter for plotting, I will be logging f(x) history there
function optimize(f, g, x0, n, prob)
    if prob == "simple1"
        x_history = optimizer_momentum_decay(f, g, x0, n; alpha = 0.2, beta = 0.8, gamma = 0.85)
    elseif prob == "simple2"
        x_history = optimizer_momentum_decay(f, g, x0, n; alpha = 1, beta = 0.9, gamma = 0.85)
    elseif prob == "simple3"
        x_history = optimizer_momentum_decay(f, g, x0, n; alpha = 1, beta = 0.9, gamma = 0.85)
    elseif prob == "secret1"
        x_history = optimizer_momentum_decay(f, g, x0, n; alpha = 1, beta = 0.9, gamma = 0.85)
    elseif prob == "secret2"
        x_history = optimizer_momentum_decay(f, g, x0, n; alpha = 1, beta = 0.9, gamma = 0.85)
    end
    x_best = last(x_history)
    return x_best
end

function optimizer_grad_descent(f, g, x0, n, alpha = 0.01, beta = 0.0001)
    x_history = [x0]

    while count(f, g) < n

        x = x_history[end]
        # f_x = f(x)
        g_x = g(x)

        # Calculate local descent
        d = -g_x / norm(g_x)

        # Backtracking Line search
        # while f(x + alpha * d) > f_x + beta * alpha * (g_x⋅d)
        #     alpha *= 0.5
        # end

        # Calculate next design point
        x_next = x_history[end] + alpha * d
        # x_next = x_history[end] - step * g_x

        # Add design point to history of design points
        push!(x_history, x_next)
    end

    return x_history
end

function optimizer_momentum(f, g, x0, n, alpha = 0.01, beta = 0)
    x_history = [x0]
    v = zeros(length(x0))

    while count(f, g) < n
        # Calculate local descent
        g_x = g(x_history[end])
        d = -g_x / norm(g_x)
        v[:] = beta*v + alpha * d

        # Calculate next design point
        x_next = x_history[end] + v

        # Add design point to history of design points
        push!(x_history, x_next)
    end

    return x_history
end

function optimizer_momentum_decay(f, g, x0, n; alpha = 0.1, beta = 0.9, gamma = 0.75)
    x_history = [x0]
    v = zeros(length(x0))

    while count(f, g) < n
        # Calculate gradient
        g_x = g(x_history[end] + beta * v)

        # Calculate local descent
        d = -g_x / norm(g_x)

        # Calculate momentum
        v[:] = beta * v + alpha * d

        # Calculate next design point
        x_next = x_history[end] + v

        # Add design point to history of design points
        push!(x_history, x_next)

        # Decay the learning rate each step
        alpha *= gamma
    end

    return x_history
end

function optimizer_ADAM(f, g, x0, n; alpha = .001, gamma_v = 0.9, gamma_s = 0.999, eps = 1E-8)
    x_history = [x0]
    v = zeros(length(x0))
    s = zeros(length(x0))

    while count(f, g) < n
        # Calculate gradient
        g_x = g(x_history[end])

        # Update biased decaying momentum
        v[:] = gamma_v * v + (1 - gamma_v) * g_x

        # Update biased decaying square gradient
        s[:] = gamma_s * s + (1 - gamma_s) * (g_x .* g_x)

        # Calculate corrected decaying momentum
        v_c = v ./ (1 - gamma_v)

        # Calculate corrected decaying square gradient
        s_c = s ./ (1 - gamma_s)

        # Calculate next design point
        x_next = x_history[end] - (alpha * v_c) ./ (sqrt.(s_c) .+ eps)

        # Add design point to history of design points
        push!(x_history, x_next)
    end

    return x_history
end