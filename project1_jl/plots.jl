using LaTeXStrings
using Plots

function rosenbrock(x, y)
    return (1.0 - x)^2 + 100.0 * (y - x^2)^2
end

function rosenbrock_gradient(x::Vector)
    storage = zeros(2)
    storage[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    storage[2] = 200.0 * (x[2] - x[1]^2)
    return storage
end

function himmelblau(x::Vector)
    return (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
end

function himmelblau_gradient(x::Vector)
    storage = zeros(2)
    storage[1] = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
        44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
    storage[2] = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
        4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
    return storage
end

function powell(x::Vector)
    return (x[1] + 10.0 * x[2])^2 + 5.0 * (x[3] - x[4])^2 +
        (x[2] - 2.0 * x[3])^4 + 10.0 * (x[1] - x[4])^4
end

function powell_gradient(x::Vector)
    storage = zeros(4)
    storage[1] = 2.0 * (x[1] + 10.0 * x[2]) + 40.0 * (x[1] - x[4])^3
    storage[2] = 20.0 * (x[1] + 10.0 * x[2]) + 4.0 * (x[2] - 2.0 * x[3])^3
    storage[3] = 10.0 * (x[3] - x[4]) - 8.0 * (x[2] - 2.0 * x[3])^3
    storage[4] = -10.0 * (x[3] - x[4]) - 40.0 * (x[1] - x[4])^3
    return storage
end






function optimizer_simple(f, g, x0, n, step = 0.01)
    x_history = [x0]
    f_history = [f(x0)]

    for i = 1:3:n-3
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

# Rosenbrock function
x_history_1, f_history_1 = optimizer_simple(rosenbrock, rosenbrock_gradient, [-1.0, -1.0], 20, 0.24)
# x_history, f_history = optimizer_momentum(rosenbrock, rosenbrock_gradient, [-1.0, -1.0], 20, 1, 0.2)

# Himmelblau function
x_history_2, f_history_2 = optimizer_simple(himmelblau, himmelblau_gradient, [-1.0, -1.0], 40, 0.3)

# Powell function
x_history_3, f_history_3 = optimizer_simple(powell, powell_gradient, [-1.0, -1.0, -1.0, -1.0], 100, 0.1)

# Plotting for rosenbrock function
# Contour of rosenbrock
xr = -2:0.1:2
yr = -2:0.1:2
contour(xr, yr, rosenbrock, levels = [10, 25, 50, 100, 200, 250, 300], colorbar = false,
        c = cgrad(:viridis, rev = true), legend = false, xlims = (-2, 2), ylims = (-2, 2),
        xlabel = "x₁", ylabel = "x₂", aspectratio = :equal, clim = (2, 500))
plot!([x_history_1[i][1] for i = 1:length(x_history_1)], [x_history_1[i][2] for i = 1:length(x_history_1)], color = :black)
savefig("Rosenbrock_Contour")

# Convergence plot
using Plots
plot(collect(1:length(f_history_1)), f_history_1, xlabel = "Iteration", ylabel = "f(x)")
savefig("Rosenbrock_Convergence")

# Convergence plot
using Plots
plot(collect(1:length(f_history_2)), f_history_2, xlabel = "Iteration", ylabel = "f(x)")
savefig("Himmelblau_Convergence")

# Convergence plot
using Plots
plot(collect(1:length(f_history_3)), f_history_3, xlabel = "Iteration", ylabel = "f(x)")
savefig("Powell_Convergence")
