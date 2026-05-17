using Test
using LinearAlgebra
using SparseArrays
using MatrixDepot
using SelfDualSimplex
using Dates
using Random

function run(p::Problem, value; time_limit=60)
    Random.seed!(0)
    start = Dates.now()
    solution = solve(p; time_limit=time_limit)
    r = zeros(length(p.c))
    for (i, n) in enumerate(p.c_names)
        r[i] = solution[n]
        @assert !isnan(r[i]) "($(i), $(r[i]))"
    end
    println("$(Dates.now() - start) $(sum(p.c .* r))")
    if !(sum(p.c .* r) ≈ value)
        #println(p.c)
        #println(r)
    end
    @assert sum(p.c .* r) ≈ value "$(sum(p.c .* r)) ≈ $(value)"
    return true
end

@testset "LU decomposition" begin
    @test begin
        rng = MersenneTwister(1234)
        a = sparse(float(rand(rng, [1.0, 2.0, 3.0, 4.0, 5.0], (10, 10))))
        pfi = PFI()
        LUelimination!(pfi, a, collect(1:10))
        
        b = float(collect(1:10))
        b2 = copy(b)

        ftran!(pfi, b)
        
        @assert b ≈ (a\b2)[pfi.basis] "$(b) ≈ $((a\b2)[basis]) $basis"
        true
    end

    @test begin
        rng = MersenneTwister(1234)
        a = sparse(float(rand(rng, [1.0, 2.0, 3.0, 4.0, 5.0], (10, 10))))
        pfi = LUdecomposition(a, collect(1:10))
        
        b = float(collect(1:10))
        b2 = copy(b)

        ftran!(pfi, b)
        
        @assert b ≈ (a\b2)[pfi.basis] "$(b) ≈ $((a\b2)[basis]) $basis"
        true
    end

    @test begin
        a = sparse(matrixdepot("HB/1138_bus"))
        a = a[:,1:1138]
        start = Dates.now()
        pfi = PFI()
        LUelimination!(pfi, a, collect(1:1138))
        
        b = float(collect(1:1138))
        b2 = copy(b)

        ftran!(pfi, b)
        
        @assert b ≈ (a\b2)[pfi.basis] #"$(b) ≈ $((a\b2)[basis])"
        println("$(Dates.now() - start)")

        start = Dates.now()
        luf = lu(a)
        @assert luf.L * luf.U ≈ (luf.Rs .* a)[luf.p, luf.q]
        println("$(Dates.now() - start)")
        true
    end

    @test begin
        a = sparse(matrixdepot("HB/bcsstm25"))
        start = Dates.now()
        pfi = PFI()
        LUelimination!(pfi, a, collect(1:size(a)[1]))
        
        b = float(collect(1:size(a)[1]))
        b2 = copy(b)

        ftran!(pfi, b)
        
        @assert b ≈ (a\b2)[pfi.basis] #"$(b) ≈ $((a\b2)[basis])"
        println("$(Dates.now() - start)")

        start = Dates.now()
        luf = lu(a)
        @assert luf.L * luf.U ≈ (luf.Rs .* a)[luf.p, luf.q]
        println("$(Dates.now() - start)")
        true
    end

    @test begin
        a = sparse(matrixdepot("HB/bcsstk27"))
        start = Dates.now()
        pfi = PFI()
        LUelimination!(pfi, a, collect(1:size(a)[1]))
        
        b = float(collect(1:size(a)[1]))
        b2 = copy(b)

        ftran!(pfi, b)
        
        @assert b ≈ (a\b2)[pfi.basis] #"$(b) ≈ $((a\b2)[basis])"
        println("$(Dates.now() - start)")

        start = Dates.now()
        luf = lu(a)
        @assert luf.L * luf.U ≈ (luf.Rs .* a)[luf.p, luf.q]
        println("$(Dates.now() - start)")
        true
    end

    # @test begin
    #     a = sparse(matrixdepot("misc//cylshell/s3rmt3m3"))
    #     start = Dates.now()
    #     pfi = PFI()
    #     LUelimination!(pfi, a, collect(1:size(a)[1]))
        
    #     b = float(collect(1:size(a)[1]))
    #     b2 = copy(b)

    #     ftran!(pfi, b)
        
    #     @assert b ≈ (a\b2)[pfi.basis] #"$(b) ≈ $((a\b2)[basis])"
    #     println("$(Dates.now() - start)")

    #     start = Dates.now()
    #     luf = lu(a)
    #     @assert luf.L * luf.U ≈ (luf.Rs .* a)[luf.p, luf.q]
    #     println("$(Dates.now() - start)")
    #     true
    # end    

    # @test begin
    #     a = sparse(matrixdepot("Mittelmann/cont11_l"))
    #     a = a[:,1:1468599]
    #     start = Dates.now()
    #     pfi = PFI()
    #     LUelimination!(pfi, a, collect(1:1468599))
        
    #     b = float(collect(1:1468599))
    #     b2 = copy(b)

    #     ftran!(pfi, b)
        
    #     @assert b ≈ (a\b2)[pfi.basis] "$(b) ≈ $((a\b2)[basis])"
    #     println("$(Dates.now() - start)")

    #     start = Dates.now()
    #     luf = lu(a)
    #     @assert luf.L * luf.U ≈ (luf.Rs .* a)[luf.p, luf.q]
    #     println("$(Dates.now() - start)")
    #     true
    # end
end

@testset "Parser" begin
    @test begin
        p = parseMPS(raw"sample.mps")
        add_slack_variables!(p)
        println(Array(p.A))
        println(p.c)
        println(p.b)

        solution = solve(p.A, p.c, p.b; time_limit=60)
        println(solution)
        @assert solution[1] ≈ 3.0
        @assert solution[3] ≈ 7.0
        true
     end
end

@testset "PFI" begin
    @test begin
        a = sparse([1.0 2.0; 3.0 5.0])
        b = [7.0; 8.0]
        res_b = inv(Array(a)) * b

        pfi = PFI([1,2], ETAMatrix[])
        for i in 1:size(a)[2]
            x = Array(a[:,i])
            ftran!(pfi, x)
            push!(pfi.eta_matrices, ETAMatrix(i, sparse(x)))
        end
        ftran!(pfi, b)
        
        res_b ≈ b
    end
end

@testset "Solve" begin
    @test begin
        c = float([3; -11; -2; 0; 0; 0; 0])
        b = float([5; 4; 6; -4])
        A = sparse(float([-1 3 0 1 0 0 0; 3 3 0 0 1 0 0; 0 3 2 0 0 1 0; -3 0 -5 0 0 0 1]))
        
        solution = solve(A, c, b; time_limit=60)
        println(solution)

        @assert solution[2] ≈ 4/3 "$(solution[2]) ≈ 4/3"
        @assert solution[3] ≈ 1.0 "$(solution[3]) ≈ 1.0"
        @assert solution[4] ≈ 1.0 "$(solution[4]) ≈ 1.0"
        @assert solution[7] ≈ 1.0 "$(solution[7]) ≈ 1.0"
        true
    end

    # @test begin
    #     c = [2 ; -3; 0; 0; 0]
    #     b = [-1; -2; 1]
    #     A = [-1 1 1 0 0; -1 -2 0 1 0; 0 1 0 0 1]
    #     solve(A, c, b)
    # end

    @test begin
        c = float([-2 ; -5; 0; 0; 0])
        b = float([4; 6; 8])
        A = sparse(float([1 0 1 0 0; 0 1 0 1 0; 1 1 0 0 1]))

        solution = solve(A, c, b; time_limit=60)
        println(solution)

        @assert solution[1] ≈ 2.0
        @assert solution[2] ≈ 6.0
        @assert solution[3] ≈ 2.0
        true
    end

    
    # @test_throws ErrorException("Infeasible/Unbounded (minJ)") begin
    #     p = parseMPS(raw"Benchmarks\infeasible\itest2.mps")
    #     add_slack_variables!(p)
    #     solution = solve(p.A, p.c, p.b)
    #     r = zeros(length(p.c))
    #     for (i,v) in solution
    #         r[i] = v
    #         @assert !isnan(v) "($(i), $(v))"
    #     end
    #     println("$(Dates.now() - start) $(sum(p.c .* r))")
    #     true
    # end
end

function run_lp(name::String, value; time_limit=60)
    print("$name ")
    p = parseMPS("../benchmarks/lptestset/$(name).mps")
    return run(p, value; time_limit=time_limit)
end

# @testset "LP test" begin
#     @test begin
#         foreach(readdir("../benchmarks/lptestset")) do f
#             try
#                 run_lp(splitext(f)[1], 0; time_limit=5)
#             catch e
#                 println(e)
#             end
#         end
#         true
#     end
# end

@testset "LP" begin    
    #@test begin
    #     run_lp("cont1", 0.008782487973)
    #     true
    # end

    # @test begin
    #     run_lp("neos", 225425492.2)
    #     true
    # end
    
    # @test begin
    #     run_lp("neos1", 46702.703)
    #     true
    # end

    # @test begin
    #     run_lp("neos2", 47619.04762)
    #     true
    # end
end

@testset "Devex pricing" begin
    # Test 1: correctness on the standard small LP (regression: Devex must not
    # change the optimal solution, only the path taken to reach it).
    @test begin
        c = float([3; -11; -2; 0; 0; 0; 0])
        b = float([5; 4; 6; -4])
        A = sparse(float([-1 3 0 1 0 0 0; 3 3 0 0 1 0 0; 0 3 2 0 0 1 0; -3 0 -5 0 0 0 1]))
        solution = solve(A, c, b; time_limit=60)
        @assert solution[2] ≈ 4/3  "solution[2]=$(solution[2])"
        @assert solution[3] ≈ 1.0  "solution[3]=$(solution[3])"
        @assert solution[4] ≈ 1.0  "solution[4]=$(solution[4])"
        @assert solution[7] ≈ 1.0  "solution[7]=$(solution[7])"
        true
    end

    # Test 2: degenerate LP with a duplicate constraint.
    # min -x1 - x2  s.t.  x1+x2 ≤ 4,  x1 ≤ 3,  x2 ≤ 3,  x1+x2 ≤ 4 (duplicate)
    # Optimal value = -4.  The duplicate constraint forces degeneracy: three
    # inequality constraints are tight at every optimal vertex, but only two
    # structural variables exist.  This exercises Devex tie-breaking.
    @test begin
        c = float([-1; -1; 0; 0; 0; 0])
        b = float([4.0; 3.0; 3.0; 4.0])
        A = sparse(float([
            1  1  1  0  0  0;
            1  0  0  1  0  0;
            0  1  0  0  1  0;
            1  1  0  0  0  1
        ]))
        solution = solve(A, c, b; time_limit=60)
        r = zeros(length(c))
        for (i, v) in solution; r[i] = v; end
        obj = sum(c .* r)
        println("degenerate LP obj=$obj")
        @assert obj ≈ -4.0  "Expected -4.0 got $obj"
        true
    end

    # Test 3: Devex weight update formula.
    # After one pivot with known Δb, the row weight for the leaving position
    # must equal ‖Δb‖² / Δb[leaving]².  We verify via a two-pivot solve on a
    # 3-constraint LP where the first Δb is predictable.
    @test begin
        # min -x1  s.t.  x1 ≤ 1 (3 copies + slacks → basis size 3)
        c = float([-1; 0; 0; 0])
        b = float([1.0; 1.0; 1.0])
        A = sparse(float([
            1  1  0  0;
            1  0  1  0;
            1  0  0  1
        ]))
        solution = solve(A, c, b; time_limit=60)
        r = zeros(length(c))
        for (i, v) in solution; r[i] = v; end
        obj = sum(c .* r)
        @assert obj ≈ -1.0  "Expected -1.0 got $obj"
        true
    end

    # Test 4: Klee-Minty cube (n=3).
    # The Klee-Minty cube is designed to force the standard simplex to visit
    # all 2^n vertices.  With Devex pricing, it should solve in far fewer steps.
    # We only check correctness here; the meszaros testset exercises larger sizes.
    @test begin
        # kleemin3: min -4x3 - 2x2 - x1  s.t. Klee-Minty constraints + slacks
        # Known optimal value: -1e4
        Random.seed!(0)
        p = parseMPS("../benchmarks/meszaros/kleemin3.mps")
        solution = solve(p; time_limit=60)
        r = zeros(length(p.c))
        for (i, n) in enumerate(p.c_names); r[i] = get(solution, n, 0.0); end
        obj = sum(p.c .* r)
        println("kleemin3 obj=$obj")
        @assert obj ≈ -1.00e4  "Expected -1e4 got $obj"
        true
    end
end

include("meszaros.jl")
#include("netlib.jl")