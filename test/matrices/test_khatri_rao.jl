using FunctionalGPs
using FunctionalGPs: KhatriRaoMatrix
using LinearAlgebra

@testset "KhatriRaoMatrix" begin
    # Factor matrices for testing
    A = [1.0 2.0 3.0; 4.0 5.0 6.0]         # 2×3
    B = [0.5 1.5 2.5; 0.1 0.2 0.3; 0.7 0.8 0.9]  # 3×3

    @testset "Column-wise (Axis=1)" begin
        # Factors share column count (n=3). Result: (2*3 × 3) = (6 × 3)
        K = KhatriRaoMatrix{1}([A, B])

        @testset "size" begin
            @test size(K) == (6, 3)
            @test size(K, 1) == 6
            @test size(K, 2) == 3
        end

        @testset "Matrix materialization" begin
            M = Matrix(K)
            @test size(M) == (6, 3)
            # Each column j: kron(B[:, j], A[:, j])
            for j in 1:3
                @test M[:, j] ≈ kron(B[:, j], A[:, j])
            end
        end

        @testset "getindex" begin
            M = Matrix(K)
            for i in 1:6, j in 1:3
                @test K[i, j] ≈ M[i, j]
            end
        end

        @testset "mul! (N=2)" begin
            v = [1.0, 2.0, 3.0]
            y = K * v
            y_ref = Matrix(K) * v
            @test y ≈ y_ref
        end

        @testset "mul! with matrix RHS" begin
            V = [1.0 0.5; 2.0 1.5; 3.0 2.5]
            Y = K * V
            Y_ref = Matrix(K) * V
            @test Y ≈ Y_ref
        end
    end

    @testset "Row-wise (Axis=2)" begin
        # Factors share row count (m=2). C is 2×3, D is 2×2 → result 2 × 6
        C = [1.0 2.0 3.0; 4.0 5.0 6.0]         # 2×3
        D = [0.5 1.5; 0.1 0.2]                  # 2×2

        K = KhatriRaoMatrix{2}([C, D])

        @testset "size" begin
            @test size(K) == (2, 6)
            @test size(K, 1) == 2
            @test size(K, 2) == 6
        end

        @testset "Matrix materialization" begin
            M = Matrix(K)
            @test size(M) == (2, 6)
            # Each row i: kron(D[i, :], C[i, :])
            for i in 1:2
                @test M[i, :] ≈ kron(D[i, :], C[i, :])
            end
        end

        @testset "getindex" begin
            M = Matrix(K)
            for i in 1:2, j in 1:6
                @test K[i, j] ≈ M[i, j]
            end
        end

        @testset "mul! (N=2)" begin
            v = rand(6)
            y = K * v
            y_ref = Matrix(K) * v
            @test y ≈ y_ref
        end

        @testset "mul! with matrix RHS" begin
            V = rand(6, 3)
            Y = K * V
            Y_ref = Matrix(K) * V
            @test Y ≈ Y_ref
        end
    end

    @testset "Three factors (N=3), column-wise" begin
        F1 = rand(2, 4)
        F2 = rand(3, 4)
        F3 = rand(2, 4)
        K = KhatriRaoMatrix{1}([F1, F2, F3])

        @test size(K) == (12, 4)

        M = Matrix(K)
        # Each column j: kron(F3[:, j], kron(F2[:, j], F1[:, j]))
        for j in 1:4
            @test M[:, j] ≈ kron(F3[:, j], kron(F2[:, j], F1[:, j]))
        end

        # mul! correctness
        v = rand(4)
        @test K * v ≈ M * v
    end

    @testset "Three factors (N=3), row-wise" begin
        F1 = rand(5, 2)
        F2 = rand(5, 3)
        F3 = rand(5, 2)
        K = KhatriRaoMatrix{2}([F1, F2, F3])

        @test size(K) == (5, 12)

        M = Matrix(K)
        # Each row i: kron(F3[i, :], kron(F2[i, :], F1[i, :]))
        for i in 1:5
            @test M[i, :] ≈ kron(F3[i, :], kron(F2[i, :], F1[i, :]))
        end

        # mul! correctness
        v = rand(12)
        @test K * v ≈ M * v
    end

    @testset "Single factor degeneracy" begin
        F = rand(3, 4)
        K1 = KhatriRaoMatrix{1}([F])
        @test Matrix(K1) ≈ F
        v = rand(4)
        @test K1 * v ≈ F * v

        K2 = KhatriRaoMatrix{2}([F])
        @test Matrix(K2) ≈ F
        v2 = rand(4)
        @test K2 * v2 ≈ F * v2
    end

    @testset "Type stability" begin
        F1 = rand(Float64, 3, 4)
        F2 = rand(Float64, 2, 4)
        K = KhatriRaoMatrix{1}([F1, F2])
        @test eltype(K) == Float64
    end

    @testset "Adjoint / transpose" begin
        A = rand(2, 3)
        B = rand(3, 3)
        K1 = KhatriRaoMatrix{1}([A, B])
        M1 = Matrix(K1)

        @testset "Column-wise adjoint indexing" begin
            K1t = K1'
            for i in 1:size(M1, 2), j in 1:size(M1, 1)
                @test K1t[i, j] ≈ M1[j, i]
            end
        end

        @testset "Column-wise adjoint * vector" begin
            v = rand(size(K1, 1))
            @test Matrix(K1') * v ≈ M1' * v
        end

        C = rand(5, 2)
        D = rand(5, 3)
        K2 = KhatriRaoMatrix{2}([C, D])
        M2 = Matrix(K2)

        @testset "Row-wise adjoint indexing" begin
            K2t = K2'
            for i in 1:size(M2, 2), j in 1:size(M2, 1)
                @test K2t[i, j] ≈ M2[j, i]
            end
        end

        @testset "Row-wise adjoint * vector" begin
            v = rand(size(K2, 1))
            @test Matrix(K2') * v ≈ M2' * v
        end
    end

    @testset "Mixed element types" begin
        F1 = rand(Float64, 3, 4)
        F2 = rand(Float64, 2, 4)
        K = KhatriRaoMatrix{1}([F1, F2])

        v32 = rand(Float32, 4)
        @test K * v32 ≈ Matrix(K) * v32
    end
end
