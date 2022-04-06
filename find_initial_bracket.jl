# Find initial bracket that includes a minimum
function find_initial_bracket(f, x0; s = 1e-2, k = 2.0)
    a, ya = x0, f(x0)
    b, yb = a + s, f(a + s)
    if yb > ya
        a,b = b, a
        ya, yb = yb, ya
        s = -s
    end
    while true
        c, yc = b + s, f(b + s)
        if yc > yb
            if a < c 
                return (a,c)
            else
                return (c, a)
            end
        end
        a, ya = b, yb
        b, yb = c, yc
        s *= k
    end
end