using Test
using LinearAlgebra
using SparseArrays
using MatrixDepot
using SelfDualSimplex
using Dates
using Random

function run(p::Problem, value)
    Random.seed!(0)
    start = Dates.now()
    solution = solve(p)
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

function run_m(name::String, value)
    print("$name ")
    p = parseMPS("Benchmarks\\meszaros\\$(name).mps")
    return run(p, value)    
end

function run_lp(name::String, value)
    print("$name ")
    p = parseMPS("Benchmarks\\lptestset\\$(name).mps")
    return run(p, value)
end

@testset "LU decomposition" begin
    @test begin
        rng = MersenneTwister(1234)
        a = sparse(float(rand(rng, [1.0, 2.0, 3.0, 4.0, 5.0], (10, 10))))
        pfi, basis = LUelimination(a, collect(1:10))
        
        b = float(collect(1:10))
        b2 = copy(b)

        ftran!(pfi, b)
        
        @assert b ≈ (a\b2)[basis] "$(b) ≈ $((a\b2)[basis]) $basis"
        true
    end

    @test begin
        rng = MersenneTwister(1234)
        a = sparse(float(rand(rng, [1.0, 2.0, 3.0, 4.0, 5.0], (10, 10))))
        pfi, basis = LUdecomposition(a, collect(1:10))
        
        b = float(collect(1:10))
        b2 = copy(b)

        ftran!(pfi, b)
        
        @assert b ≈ (a\b2)[basis] "$(b) ≈ $((a\b2)[basis]) $basis"
        true
    end

    @test begin
        a = sparse(matrixdepot("HB/1138_bus"))
        a = a[:,1:1138]
        start = Dates.now()
        pfi, basis = LUelimination(a, collect(1:1138))
        
        b = float(collect(1:1138))
        b2 = copy(b)

        ftran!(pfi, b)
        
        @assert b ≈ (a\b2)[basis] #"$(b) ≈ $((a\b2)[basis])"
        println("$(Dates.now() - start)")

        start = Dates.now()
        luf = lu(a)
        @assert luf.L * luf.U ≈ (luf.Rs .* a)[luf.p, luf.q]
        println("$(Dates.now() - start)")
        true
    end

    # @test begin
    #     a = sparse(matrixdepot("Mittelmann/cont11_l"))
    #     a = a[:,1:1468599]
    #     start = Dates.now()
    #     pfi, basis = LUelimination(a, collect(1:1468599))
        
    #     b = float(collect(1:1468599))
    #     b2 = copy(b)

    #     ftran!(pfi, b)
        
    #     @assert b ≈ (a\b2)[basis] "$(b) ≈ $((a\b2)[basis])"
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

        pfi = PFI(ETAMatrix[])
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

    
    @test_throws ErrorException("Infeasible/Unbounded (minJ)") begin
        p = parseMPS(raw"Benchmarks\infeasible\itest2.mps")
        add_slack_variables!(p)
        solution = solve(p.A, p.c, p.b)
        r = zeros(length(p.c))
        for (i,v) in solution
            r[i] = v
            @assert !isnan(v) "($(i), $(v))"
        end
        println("$(Dates.now() - start) $(sum(p.c .* r))")
        true
    end
end

@testset "Meszaros" begin
    @test begin
        run_m("kleemin3", -1.00e4)
        run_m("kleemin4", -1.00e6)
        run_m("kleemin5", -1.00e8)
        run_m("kleemin6", -1.00e10)
        run_m("kleemin7", -1.00e12)
        run_m("kleemin8", -1.00e14)
        true
    end

    @test begin
        run_m("aa01", 55535.43639)
        run_m("aa03", 49616.36356)
        run_m("aramco", -3.926918e5)
        run_m("air02", 7640)
        run_m("air03", 338864.25)
        run_m("air04", 55535.43639)
        run_m("air05", 25877.60927)
        #run_m("air06", 49616.36287)
        run_m("aircraft", 1567.042349)
        run_m("baxter_mat", 56007255.67)
        run_m("jendrec1", 7028.460511)
        run_m("lindo", 1.750000e4)
        run_m("model1", 0.0)
        run_m("nsct1", -3.8922436000e7)
        run_m("nsct2", -3.7175082000e7)
        run_m("nw14", 6.1844000000e4)
        run_m("p0033", 2520.571739)
        run_m("p0040", 61796.54505)
        run_m("p0201", 6875.0)
        run_m("p0282", 176867.5033)
        run_m("p12345", -42122.67947)
        run_m("pcb1000", 56809.45689)
        run_m("pcb3000", 137416.4196)
        run_m("seymour", 403.8464741)
        run_m("test", -2351871.325)
        run_m("zed", -15060.64524)
        #run_m("model2", -7.400489e3)
        #run_m("delfland", 2.203089e0)
        #run_m("p19", 253964.3546)
        #run_m("dsbmip", -305.20)
        #run_m("f2177", 90.00)
        true
    end

    @test begin
        run_m("nsic1", -9.168554e6)
        run_m("nsic2", -8.203512e6)
        true
    end

    # @test begin
    #     run_m("dbir1", -8.1067070000e6)
    #     run_m("dbir2", -6.1169165000e6)
    #     true
    # end

    # @test begin
    #     run_m("radio", 0.0)
    #     true
    # end
    # @test begin
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

