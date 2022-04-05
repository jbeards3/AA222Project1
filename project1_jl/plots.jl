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

function optimizer(f, g, x0, n, step)
    x_history = [x0]
    f_history = [f(x0)]

    for i = 1:n-1
        x_next = x_history[end] + step * ones(length(x0))

        push!(x_history, x_next)
        push!(f_history, f(x_next))
    end

    return x_history, f_history
end

# Plotting for rosenbrock function
x_history, f_history = optimizer(rosenbrock, rosenbrock_gradient, [-1.0, -1.0], 10, 0.1)

## Contour of rosenbrock
xr = -2:0.1:2
yr = -2:0.1:2
contour(xr, yr, rosenbrock, levels = [10, 25, 50, 100, 200, 250, 300], colorbar = false,
        c = cgrad(:viridis, rev = true), legend = false, xlims = (-2, 2), ylims = (-2, 2),
        xlabel = "x₁", ylabel = "x₂", aspectratio = :equal, clim = (2, 500))
plot!([x_history[i][1] for i = 1:length(x_history)], [x_history[i][2] for i = 1:length(x_history)], color = :black)
savefig("Rosenbrock_Contour")

## Convergence plot
using Plots
plot(collect(1:length(f_history)), f_history, xlabel = "Iteration", ylabel = "f(x)")
savefig("Rosenbrock_Convergence")
