export covfunc_integral_one_sided_quad, covfunc_integral_two_sided_quad

covfunc_integral_one_sided_quad(k, x, a, b) = quadgk(y -> k(x, y), a, b)[1]
function covfunc_integral_two_sided_quad(k, a, b, c, d)
    return quadgk(x -> covfunc_integral_one_sided_quad(k, x, c, d), a, b)[1]
end
