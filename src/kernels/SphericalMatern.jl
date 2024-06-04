module MaternKernelModule

using SpecialFunctions
using LinearAlgebra
using Polynomials
using LegendrePolynomials

export MaternKernel, ExtrinsicMatern, IntrinsicMatern, evaluate, legendre_polys, angles_to_r3

struct MaternKernel
    nu::Float64 ## Smoothness parameter
    rho::Float64 ## Lengthscale parameter
    var::Float64 ## Variance parameter
end

struct ExtrinsicMatern
    matern_kernel::MaternKernel
end

struct IntrinsicMatern
    matern_kernel::MaternKernel
end

# Note: Im directly using the latitude and longitude angles as input, compared to the formatting it as x1 & x2 tuples in the python code

# Evaluation method for ExtrinsicMatern
function evaluate(em::ExtrinsicMatern, θ1::Float64, φ1::Float64, θ2::Float64, φ2::Float64)
    x1 = angles_to_r3(θ1, φ1)
    x2 = angles_to_r3(θ2, φ2)
    return evaluate(em.matern_kernel, x1, x2)
end

# Evaluation method for IntrinsicMatern
# different to python code: kappa is renamed into rho
function evaluate(im::IntrinsicMatern, θ1::Float64, φ1::Float64, θ2::Float64, φ2::Float64, truncation=nothing)

    if truncation == nothing
        truncation = 30
    end

    x1 = angles_to_r3(θ1, φ1)
    x2 = angles_to_r3(θ2, φ2)

    dist = dot(x1, x2) # Compute inner product

    k = 0.0

    polys = legendre_polys(truncation)

    frac_2nu_ksq = 2 * im.matern_kernel.nu / (im.matern_kernel.rho)^2
    exponent = -(im.matern_kernel.nu + 1)
    
    for n in 0:truncation-1
        k += (frac_2nu_ksq + n*(n+1))^exponent * (2*n + 1) * polys[n+1](dist)
        #println("polys: ", polys[n+1])
        #println("dist: ", dist)
        #println("evaluated: ", polys[n+1](dist))
    end
    
    k *= im.matern_kernel.var / (4 * π)
    return k

end

# Evaluation method for MaternKernel
# also included rho, compared to the python implementation
function evaluate(kernel::MaternKernel, x1::Vector{Float64}, x2::Vector{Float64})
    r = norm(x1 - x2)/kernel.rho * sqrt(2 * kernel.nu)
    return (2^(1 - kernel.nu) / gamma(kernel.nu)) * (r^kernel.nu) * besselk(kernel.nu, r)
end


# done, works properly (checked against python code)
function legendre_polys(degree::Int64)
    if degree < 1
        throw(ArgumentError("The degree of the Legendre polynomials must be positive."))
    end
    
    polys = Vector{Polynomial}(undef, degree)
    polys[1] = Polynomial([1.0]) # degree 0 Legendre polynomial
    if degree >= 2
        polys[2] = Polynomial([0.0, 1.0]) # degree 1 Legendre polynomial
    
        for n in 1:degree-2
            polys[n+2] = (2*n+1)/(n+1) * (polys[n+1] * Polynomial([0.0, 1.0])) - n/(n+1) * polys[n] # recursion relation
        end
    end
    
    return polys
end

#print(legendre_polys(5))

# Convert angles (θ, φ) to points in R^3
function angles_to_r3(θ::Float64, φ::Float64)
    x = cos(φ) * sin(θ)
    y = sin(φ) * sin(θ)
    z = cos(θ)
    return [x, y, z]
end

end