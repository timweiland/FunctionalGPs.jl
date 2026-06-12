using Luxor
using FunctionalGPs
using AbstractGPs
using LinearAlgebra

# --- GP Setup ---
kernel = 0.25 * WendlandKernel(2, 3, 1.2)  # larger lengthscale for smoother GP
f = GP(kernel)

# Observation points (y values scaled to ~unit scale)
x_obs = [0.15, 0.5, 0.85]
y_obs = [0.7, 1.1, 0.7]

# Condition on observations (more noise = doesn't interpolate exactly)
f_post = condition_on_observation(f, x_obs, y_obs; noise = 0.005)

# Integral observation (for the shaded region on the left)
int_start, int_end = 0.22, 0.42
int_domain = [Interval(int_start, int_end)]
int_functional = VectorizedLebesgueIntegral(int_domain)
# Integral value chosen to be consistent with the curve shape
f_post2 = condition_on_observation(f_post, int_functional, [0.12]; noise = 3.0e-4)

# Derivative observation (for tangent line) - on the right side
x_deriv = 0.675
dx = PartialDerivative((1,))
deriv_functional = EvaluationFunctional([x_deriv]) ∘ dx
f_post3 = condition_on_observation(f_post2, deriv_functional, [-1.0]; noise = 1.0e-6)

# --- Compute posterior ---
x_plot = collect(range(0.02, 0.98, length = 200))
μ, v = mean_and_var(f_post3(x_plot))
σ = sqrt.(v)

# Get derivative value at tangent point for drawing
tangent_idx = argmin(abs.(x_plot .- x_deriv))
tangent_y = μ[tangent_idx]
tangent_slope = -1.0  # We conditioned on this

# --- Drawing parameters ---
width, height = 400, 400
margin = 40
plot_width = width - 2 * margin
plot_height = height - 2 * margin

# Data range for scaling
y_min, y_max = 0.0, 1.8

# Transform data coordinates to drawing coordinates
function to_canvas(x, y)
    cx = margin + x * plot_width
    cy = height - margin - (y - y_min) / (y_max - y_min) * plot_height
    return Point(cx, cy)
end

# Julia colors
julia_purple = (0.584, 0.345, 0.698)
julia_green = (0.22, 0.596, 0.149)
julia_red = (0.796, 0.235, 0.2)
julia_blue = (0.251, 0.388, 0.847)

# --- Generate SVG ---
Drawing(width, height, "/Users/chelmney/.julia/dev/FunctionalGPs/docs/src/assets/logo.svg")
origin(Point(0, 0))
background("transparent")

# Draw uncertainty band
upper_points = [to_canvas(x_plot[i], μ[i] + 2σ[i]) for i in eachindex(x_plot)]
lower_points = [to_canvas(x_plot[i], μ[i] - 2σ[i]) for i in reverse(eachindex(x_plot))]
band_points = vcat(upper_points, lower_points)

setcolor(julia_purple..., 0.25)
poly(band_points, :fill, close = true)

# Draw integral region (on the left side)
int_start, int_end = 0.22, 0.42
int_indices = findall(x -> int_start <= x <= int_end, x_plot)

if !isempty(int_indices)
    int_upper = [to_canvas(x_plot[i], μ[i]) for i in int_indices]
    int_lower = [to_canvas(x_plot[i], y_min) for i in reverse(int_indices)]
    int_poly = vcat(int_upper, int_lower)

    setcolor(julia_green..., 0.4)
    poly(int_poly, :fill, close = true)

    # Integral bounds (dashed lines)
    setcolor(julia_green...)
    setdash("shortdashed")
    setline(2.5)
    line(to_canvas(int_start, y_min), to_canvas(int_start, μ[int_indices[1]]), :stroke)
    line(to_canvas(int_end, y_min), to_canvas(int_end, μ[int_indices[end]]), :stroke)
    setdash("solid")
end

# Draw mean curve
mean_points = [to_canvas(x_plot[i], μ[i]) for i in eachindex(x_plot)]
setcolor(julia_purple...)
setline(5)
setlinecap("round")
setlinejoin("round")
poly(mean_points, :stroke)

# Draw tangent line
tangent_len = 0.12
t_x1 = x_deriv - tangent_len
t_x2 = x_deriv + tangent_len
t_y1 = tangent_y - tangent_slope * tangent_len
t_y2 = tangent_y + tangent_slope * tangent_len

setcolor(julia_red...)
setline(4)
line(to_canvas(t_x1, t_y1), to_canvas(t_x2, t_y2), :stroke)

# Tangent point
#setcolor(julia_red...)
#circle(to_canvas(x_deriv, tangent_y), 8, :fill)

# Draw observation points
setcolor(julia_blue...)
for (x, y) in zip(x_obs, y_obs)
    circle(to_canvas(x, y), 7, :fill)
end

finish()

println("Logo generated at docs/src/assets/logo.svg")
