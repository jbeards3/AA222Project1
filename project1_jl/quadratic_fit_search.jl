function quadratic_fit_search(f, x0, n)

    (a, c) = find_initial_bracket(f, x0)
    b = (a + c) / 2

    ya, yb, yc = f(a), f(b), f(c)

    for i = 1:n-3
        x = 0.5 * (ya * (b^2 - c^2) + yb * (c^2 - a^2) + yc * (a^2 - b^2)) / 
        (ya * (b - c) + yb * (c - a) + yc * (a - b))

        yx = f(x)

        if x > b
            if yx > yb
                c, yc = x, yx
            else
                a, ya = b, yb
                b, yb = x, yx
            end
        elseif x < b
            if yx > yb
                a, ya = x, yx
            else
                c, yc = b, yb
                b, yb = x, yx
            end
        end
    end
    return (a, b, c)
end