using LaTeXStrings
using Plots
using LinearAlgebra

## Functions
function rosenbrock_plot(x, y)
    return (1.0 - x)^2 + 100.0 * (y - x^2)^2
end

function rosenbrock(x::Vector)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
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

## Optimizer Function
function optimizer_momentum_decay(f, g, x0, n; alpha = 0.01, beta = 0, gamma = 1)
    x_history = [x0]
    f_history = [f(x0)]
    v = zeros(length(x0))

    for i = 1:2:n
        # Calculate gradient
        g_x = g(x_history[end] + beta * v)

        # Calculate local descent
        d = -g_x / norm(g_x)

        # Calculate momentumS
        v = beta*v + alpha * gamma * d

        # Decay learning rate
        alpha *= gamma

        # Calculate next design point
        x_next = x_history[end] + v

        # Add design point to history of design points
        push!(x_history, x_next)
        push!(f_history, f(x_next))
    end

    return x_history, f_history
end

## Function Minimization
# Rosenbrock function
x_history_r1, f_history_r1 = optimizer_momentum_decay(rosenbrock, rosenbrock_gradient, [-1.0, -1.0], 100; alpha = 0.2, beta = 0.8, gamma = 0.85)
x_history_r2, f_history_r2 = optimizer_momentum_decay(rosenbrock, rosenbrock_gradient, [0, -1.0], 100; alpha = 0.2, beta = 0.8, gamma = 0.85)
x_history_r3, f_history_r3 = optimizer_momentum_decay(rosenbrock, rosenbrock_gradient, [-1.0, 0], 100; alpha = 0.2, beta = 0.8, gamma = 0.85)

# Himmelblau function
x_history_h1, f_history_h1 = optimizer_momentum_decay(himmelblau, himmelblau_gradient, [-1.0, -1.0], 100; alpha = 1, beta = 0.9, gamma = 0.85)
x_history_h2, f_history_h2 = optimizer_momentum_decay(himmelblau, himmelblau_gradient, [0, -1.0], 100; alpha = 1, beta = 0.9, gamma = 0.85)
x_history_h3, f_history_h3 = optimizer_momentum_decay(himmelblau, himmelblau_gradient, [-1.0, 0], 100; alpha = 1, beta = 0.9, gamma = 0.85)

# Powell function
x_history_p1, f_history_p1 = optimizer_momentum_decay(powell, powell_gradient, [-1.0, -1.0, -1.0, -1.0],100; alpha = 1, beta = 0.9, gamma = 0.85)
x_history_p2, f_history_p2 = optimizer_momentum_decay(powell, powell_gradient, [0, 0, -1.0, -1.0], 100; alpha = 1, beta = 0.9, gamma = 0.85)
x_history_p3, f_history_p3 = optimizer_momentum_decay(powell, powell_gradient, [-1.0, -1.0, 0, 0], 100; alpha = 1, beta = 0.9, gamma = 0.85)

## Plotting
# Contour of Rosenbrock
xr = -2:0.1:2
yr = -2:0.1:2
contour(xr, yr, rosenbrock_plot, levels = [1, 5, 10, 25, 50, 100, 200, 300, 400, 500], colorbar = true,
        c = cgrad(:viridis, rev = true), legend = false, xlims = (-2, 2), ylims = (-2, 2),
        xlabel = "x₁", ylabel = "x₂", aspectratio = :equal, clim = (1, 500), title = "Contour Plot - Rosenbrock")
plot!([x_history_r1[i][1] for i = 1:length(x_history_r1)], [x_history_r1[i][2] for i = 1:length(x_history_r1)],
 color = :blue, label = "Starting Position: [-1.0, -1.0]")

plot!([x_history_r2[i][1] for i = 1:length(x_history_r2)], [x_history_r2[i][2] for i = 1:length(x_history_r2)],
 color = :red, label = "Starting Position: [0, -1.0]")
 
plot!([x_history_r3[i][1] for i = 1:length(x_history_r3)], [x_history_r3[i][2] for i = 1:length(x_history_r3)],
 color = :green, label = "Starting Position: [-1.0, 0]")

savefig("Rosenbrock_Contour")

# Convergence plot - Rosenbrock
plot(collect(1:length(f_history_r1)), f_history_r1, xlabel = "Iteration", ylabel = "f(x)",
color = :blue, label = "Starting Position: [-1.0, -1.0]", title = "Convergence Plot - Rosenbrock")

plot!(collect(1:length(f_history_r2)), f_history_r2, xlabel = "Iteration", ylabel = "f(x)",
color = :red, label = "Starting Position: [0, -1.0]")

plot!(collect(1:length(f_history_r3)), f_history_r3, xlabel = "Iteration", ylabel = "f(x)",
color = :green, label = "Starting Position: [-1.0, 0]")

savefig("Rosenbrock_Convergence")

# Convergence plot - Himmelblau
plot(collect(1:length(f_history_h1)), f_history_h1, xlabel = "Iteration", ylabel = "f(x)",
color = :blue, label = "Starting Position: [-1.0, -1.0]", title = "Convergence Plot - Himmelblau")

plot!(collect(1:length(f_history_h2)), f_history_h2, xlabel = "Iteration", ylabel = "f(x)",
color = :red, label = "Starting Position: [0, -1.0]")

plot!(collect(1:length(f_history_h3)), f_history_h3, xlabel = "Iteration", ylabel = "f(x)",
color = :green, label = "Starting Position: [-1.0, 0]")
savefig("Himmelblau_Convergence")

# Convergence plot - Powell
plot(collect(1:length(f_history_p1)), f_history_p1, xlabel = "Iteration", ylabel = "f(x)",
color = :blue, label = "Starting Position: [-1.0, -1.0, -1.0, -1.0]", title = "Convergence Plot - Powell")

plot!(collect(1:length(f_history_p2)), f_history_p2, xlabel = "Iteration", ylabel = "f(x)",
color = :red, label = "Starting Position: [0, 0, -1.0, -1.0]")

plot!(collect(1:length(f_history_p3)), f_history_p3, xlabel = "Iteration", ylabel = "f(x)",
color = :green, label = "Starting Position: [-1.0, -1.0, 0, 0]")

savefig("Powell_Convergence")
