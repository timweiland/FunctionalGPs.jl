export moveaxis, reshape_product_broadcast

function moveaxis(A::AbstractArray, source::Int, dest::Int)
    if source == dest
        return A
    end
    perm = collect(1:ndims(A))
    insert!(perm, dest, splice!(perm, source))
    return permutedims(A, perm)
end

"""
    reshape_product_broadcast(A::AbstractArray, B::AbstractArray)

    Reshape `A` and `B` by adding singleton dimensions to enable broadcast 
    operations on the Cartesian product of the two arrays.

    # Example
    ```julia
    A = rand(4, 5)
    B = rand(2, 3)
    A, B = reshape_product_broadcast(A, B)
    C = A .+ B
    @assert size(C) == (4, 5, 2, 3)
    ```
"""
function reshape_product_broadcast(A::AbstractArray, B::AbstractArray)
    ndims_A = ndims(A)
    A = reshape(A, size(A)..., ones(Int, ndims(B))...)
    B = reshape(B, ones(Int, ndims_A)..., size(B)...)
    return A, B
end
