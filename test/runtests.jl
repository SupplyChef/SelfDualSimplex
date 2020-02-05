using Test
using LinearAlgebra
using SparseArrays
using MatrixDepot
using SelfDualSimplex
using Dates
using Random

function run(p::Problem, value; time_limit=-1)
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

        solution = solve(p.A, p.c, p.b)
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
        
        solution = solve(A, c, b)
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

        solution = solve(A, c, b)
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

function run_lp(name::String, value; time_limit=-1)
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

include("meszaros.jl")
#include("netlib.jl")