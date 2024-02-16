export moveaxis

function moveaxis(A::AbstractArray, source::Int, dest::Int)
    if source == dest
        return A
    end
    perm = collect(1:ndims(A))
    insert!(perm, dest, splice!(perm, source))
    return permutedims(A, perm)
end
