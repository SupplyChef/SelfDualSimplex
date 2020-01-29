module SelfDualSimplex

import  Base.*
import  Base.size
import  Base.getindex
import  Base.setindex

using SparseArrays
using LinearAlgebra
using DelimitedFiles

include("MPSParser.jl")
include("PFI.jl")

export PFI
export ETAMatrix
export ftran!
export btran!
export solve

export parseMPS
export Problem
export add_upper_bounds!
export add_lower_bounds!
export add_free_variables!
export add_slack_variables!

export LUelimination
export LUdecomposition
export eliminate

function get_column!(c::Array{Float64, 1}, m::SparseMatrixCSC{Float64, Int64}, j::Int64)
    fill!(c, 0.0)
    @inbounds for k in m.colptr[j]:(m.colptr[j+1]-1)
        c[m.rowval[k]] = m.nzval[k]  
    end
end

function max_argmax(data)
    max_value = -Inf64
    max_index = -1
    @inbounds for i in 1:length(data)
        value = data[i]
        if value > max_value
            max_value = value
            max_index = i
        end
    end
    return (max_value, max_index)
end

function min_argmin(data)
    min_value = Inf64
    min_index = -1
    @inbounds for i in 1:length(data)
        value = data[i]
        if value < min_value
            min_value = value
            min_index = i
        end
    end
    return (min_value, min_index)
end

function computeΔc!(Δc::Array{Float64}, A::SparseMatrixCSC{Float64,Int64}, el::Array{Float64}, pfi::PFI, is_basic::Array{Bool}, l::Int64)
    c = size(A)[2]
    b = size(A)[1]

    fill!(el, 0.0)
    el[l] = 1.0
    btran!(pfi, el)
    #mul!(Δc,  A', el)
    
    @inbounds @simd for i in 1:c
        # if is_basic[i]
        #     Δc[i] = 0.0
        # else
        if i > c - b
            Δc[i] = el[i - (c - b)]
            if abs(Δc[i]) < 1e-12
                Δc[i] = 0
            end
        else 
            Δc[i] = dot(A, i, el)
            if abs(Δc[i]) < 1e-12
                Δc[i] = 0
            end
        end
    end
end

function computeΔc2!(Δc::Array{Float64}, tA::SparseMatrixCSC{Float64,Int64}, el::Array{Float64}, pfi::PFI, is_basic::Array{Bool}, l::Int64, eps::Float64)
    c = size(tA)[1]
    b = size(tA)[2]
    
    fill!(el, 0.0)
    el[l] = 1.0
    btran!(pfi, el)
    
    fill!(Δc, 0.0)
    
    for i in 1:length(el)
        if abs(el[i]) > eps
            (Is, Vs) = get_nz(tA, i)
            for j in 1:length(Is)
                Δc[Is[j]] += el[i] * Vs[j]
            end
        end
    end

    for i in 1:length(Δc)
        if i > c - b
            Δc[i] = el[i - (c - b)]
        end
        #else 
        #if is_basic[i]
        #     Δc[i] = 0
        # end
        if abs(Δc[i]) < 1e-12
            Δc[i] = 0
        end
        #end
    end
end

function add_upper_bounds!(p)
    @assert length(p.b) == size(p.A)[1] "$(length(p.b)) != $(size(p.A)[1])"
    @assert length(p.c) == size(p.A)[2]

    lb = length(p.b)
    lc = length(p.c)

    bound_count = count(p.upper_bounds .< Inf64)
    
    resize!(p.b, length(p.b) + bound_count)

    (Is, Js, Vs) = findnz(p.A)

    #p.A = resize(p.A, size(p.A)[1] + bound_count, size(p.A)[2])
    j = 1
    for i in 1:length(p.upper_bounds)
        if p.upper_bounds[i] < Inf64
            p.b[lb + j] = p.upper_bounds[i]
            push!(Is, lb + j)
            push!(Js, i)
            push!(Vs, 1.0)            
            #p.A[lb + j, i] = 1.0
            j += 1
        end
    end
    p.A = sparse(Is, Js, Vs, size(p.A)[1] + bound_count, size(p.A)[2])

    @assert length(p.b) == size(p.A)[1]
    @assert length(p.c) == size(p.A)[2]
end

function add_lower_bounds!(p)
    @assert length(p.b) == size(p.A)[1]
    @assert length(p.c) == size(p.A)[2]

    lb = length(p.b)
    lc = length(p.c)    

    bound_count = count(p.lower_bounds .> 0.0)
    
    resize!(p.b, length(p.b) + bound_count)

    (Is, Js, Vs) = findnz(p.A)
    #p.A = resize(p.A, size(p.A)[1] + bound_count, size(p.A)[2])
    j = 1
    for i in 1:length(p.lower_bounds)
        if p.lower_bounds[i] > 0.0
            p.b[lb + j] = -p.lower_bounds[i]
            push!(Is, lb + j)
            push!(Js, i)
            push!(Vs, -1.0)            
            #p.A[lb + j, i] = -1.0
            j += 1
        end
    end
    p.A = sparse(Is, Js, Vs, size(p.A)[1] + bound_count, size(p.A)[2])

    @assert length(p.b) == size(p.A)[1]
    @assert length(p.c) == size(p.A)[2]
end

function add_free_variables!(p)
    @assert length(p.b) == size(p.A)[1]
    @assert length(p.c) == size(p.A)[2]

    lc = length(p.c)

    bound_count = count((p.lower_bounds .== -Inf64) .& (p.upper_bounds .== Inf64))
    
    resize!(p.c, length(p.c) + bound_count)
    p.A = resize(p.A, size(p.A)[1], size(p.A)[2] + bound_count)
    j = 1
    for i in 1:length(p.lower_bounds)
        if p.lower_bounds[i] == -Inf64 && p.upper_bounds[i] == Inf64
            p.A[:, lc + j] = -p.A[:, i]
            p.c[lc + j] = -p.c[i]
            j += 1
        end
    end

    @assert length(p.b) == size(p.A)[1]
    @assert length(p.c) == size(p.A)[2]
end

function add_slack_variables!(p)
    @assert length(p.b) == size(p.A)[1]
    @assert length(p.c) == size(p.A)[2]

    lb = length(p.b)
    lc = length(p.c)

    resize!(p.c, length(p.c) + lb)
    #p.A = resize(p.A, size(p.A)[1], size(p.A)[2] + lb)
    (Is, Js, Vs) = findnz(p.A)
    for i in 1:lb
        p.c[lc + i] = 0.0
        push!(Is, i)
        push!(Js, lc + i)
        push!(Vs, 1.0)
        #p.A[i, lc + i] = 1.0
    end
    p.A = sparse(Is, Js, Vs, size(p.A)[1], size(p.A)[2] + lb)
end

function resize(a::SparseMatrixCSC{Float64, Int64}, m, n)
    (I, J, V) = findnz(a)
    return sparse(I, J, V, m, n)
end

function presolve!(p, presolution::Dict{String, Float64})
    # p.A * x <= p.b
    active = repeat([true], length(p.lower_bounds))
    for i in 1:length(p.lower_bounds)
        if p.lower_bounds[i] == p.upper_bounds[i]
            active[i] = false            
            if p.lower_bounds[i] != 0.0
                (Is, Vs) = get_nz(p.A, i)
                for k in 1:length(Is)
                    #println("p.b[$(Is[k])] $(p.b[Is[k]]) => $(p.b[Is[k]] - p.lower_bounds[i] * Vs[k])")
                    p.b[Is[k]] -= p.lower_bounds[i] * Vs[k]
                end
            end
            push!(presolution, p.c_names[i] => p.lower_bounds[i])
        end
    end

    p.A = p.A[:,active]
    p.c = p.c[active]
    p.upper_bounds = p.upper_bounds[active]
    p.lower_bounds = p.lower_bounds[active]
    p.c_names = p.c_names[active]

    active = repeat([true], length(p.b))
    (Is, Js, Vs) = findnz(p.A)
    tA = sparse(Js, Is, Vs, size(p.A)[2], size(p.A)[1])
    row_scaling = repeat([1.0], size(p.A)[1])
    for i in 1:size(tA)[2]
        (Is, Vs) = get_nz(tA, i)
        if length(Vs) > 0
            maxV = maximum(sqrt.(abs.(Vs)))
            if maxV > 1.0
                row_scaling[i] = 1.0 / maxV
            end
        else
            if p.b[i] < 0.0
                throw(ErrorException("Infeasible: $i $(p.b[i])"))
            end
            active[i] = false
        end
    end
    tA = tA[:,active]
    p.b = p.b[active]
    row_scaling = row_scaling[active]

    (Is, Js, Vs) = findnz(tA)
    p.A = sparse(Js, Is, Vs, size(tA)[2], size(tA)[1])
    
    d = Diagonal(row_scaling)
    p.A = d * p.A
    p.b = d * p.b
end

function solve(p)
    p = deepcopy(p)
    print("$(size(p.A)) ")
    presolution = Dict{String, Float64}()
    presolve!(p, presolution)
    print("$(size(p.A)) ")
    add_upper_bounds!(p)
    add_lower_bounds!(p)
    add_free_variables!(p)
    add_slack_variables!(p)
    print("$(size(p.A)) ")
    solution = solve(p.A, p.c, p.b)
    solution2 = Dict{String, Float64}(p.c_names[i] => get(solution, i, 0.0) for i in 1:length(p.c_names))
    merge!(solution2, presolution)
    return solution2
end

function solve(A::SparseMatrixCSC{Float64, Int64},c::Array{Float64,1},b::Array{Float64,1})
    
    @assert size(A)[1] == length(b) "Size of b and size of A do not match: $(size(A)[1]) != $(length(b))"
    @assert size(A)[2] == length(c) "Size of c and size of A do not match: $(size(A)[2]) != $(length(c))"
    @assert count(x -> isnan(x), b) == 0
    @assert count(x -> isnan(x), c) == 0

    (Is, Js, Vs) = findnz(A)
    tA = sparse(Js, Is, Vs, size(A)[2], size(A)[1])

    eps = 1e-8
    t = 0.0
        
    basis = collect(length(c) - length(b) + 1:length(c))
    is_basic = zeros(Bool, length(c))
    for b in basis
        is_basic[b] = true
    end
    pfi, basis = LUelimination(A, basis)
 
    b_hat = copy(b) # values of basic variables
    c_hat = copy(c) # costs 

    perturbation_c = vcat(repeat([1.0], length(c)-length(b)) .+ 5 .* rand(Float64, length(c)-length(b)), repeat([0.0], length(b)))
    perturbation_c_hat = copy(perturbation_c)
    perturbation_b = repeat([1.0], length(b)) .+ 5 .* rand(Float64, length(b))
    perturbation_b_hat = copy(perturbation_b)
    
    old_t = Inf64
    iter = 0

    leaving = -1
    j = -1
    el = zeros(length(b))
    Δb = zeros(length(b))
    Δc = zeros(length(c))
    Δz = zeros(length(c))

    pb = zeros(length(b))
    pc = zeros(length(c))

    forced_refactoring = false
    primal_count = 0
    dual_count = 0
    while(true)
        #@assert sum(is_basic) == length(basis)
        iter = iter + 1

        fill!(pb, 0.0)
        @inbounds for i in 1:length(b_hat)
            if b_hat[i] < 0
                violation = -b_hat[i]
                pb[i] = violation / perturbation_b_hat[i]
            end
        end
        fill!(pc, 0.0)
        @inbounds for i in 1:length(c_hat)
            if !is_basic[i] && c_hat[i] < 0 
                violation = -c_hat[i]
                pc[i] = violation / perturbation_c_hat[i]
            end
        end
        (t_b, leaving) = max_argmax(pb)
        (t_c, j) = max_argmax(pc)
        t = max(t_b, t_c)
        
        #@assert t <= old_t "$t $old_t"
        old_t = t

        # println("b_hat:$b_hat") 
        # println("c_hat:$c_hat")
        # println("perturbation_b_hat:$perturbation_b_hat") 
        # println("perturbation_c_hat:$perturbation_c_hat") 
        if iter % 15000 == 0
            r = zeros(length(c))
            for (i,v) in Dict(zip(basis, b_hat))
                r[i] = v
                @assert !isnan(v) "($(i), $(v))"
            end
            println("$iter $(sum(c .* r)) t:$(max(t_b,t_c)) t_b:$t_b t_c:$t_c primals:$primal_count duals:$dual_count") 
        end
        
        if t_b <= eps && t_c <= eps
            @debug "$iter t_b: $t_b t_c: $t_c "#b_hat: $(b_hat[leaving]) perturbation_b_hat: $(perturbation_b_hat[leaving]) c_hat: $(c_hat) perturbation_c_hat: $(perturbation_c_hat)"
            println("primal: $primal_count dual: $dual_count")
            return Dict(zip(basis, b_hat))
        end
        
        try
        #if t_b > t_c
        if t_b > t_c
            dual_count += 1
            #dual simplex step   
            #computeΔc!(Δc, A, el, pfi, is_basic, leaving)
            #Δc2 = copy(Δc)
            computeΔc2!(Δc, tA, el, pfi, is_basic, leaving, eps)
            # for i in 1:length(Δc)
            #     @assert Δc[i] ≈ Δc2[i] "$(Δc[i]) ≈ $(Δc2[i])"
            # end
            Δz = -Δc
                
            fill!(pc, Inf64)
            @inbounds for i in 1:length(c_hat)
                if !is_basic[i] && Δz[i] > eps  #c will decrease; it should not go negative for variables that are not at their upper bound (or we loose optimality)
                    @assert (c_hat[i] + t * perturbation_c_hat[i]) > 0 "iter: $iter c_hat: $(c_hat[i]) perturbation_c_hat: $(perturbation_c_hat[i]) t: $t bn_hat: $(bn_hat[i]) upper: $(upper_bounds[i]) $(c_hat[i] + t * perturbation_c_hat[i]) > 0"
                    pc[i] = (c_hat[i] + t * perturbation_c_hat[i]) / Δz[i]
                end
            end
            (minJ, j) = min_argmin(pc)
            if isinf(minJ)
                @debug "$iter t_b: $t_b t_c: $t_c entering:? leaving:$(basis[leaving]) b_hat: $(b_hat[leaving]) perturbation_b_hat: $(perturbation_b_hat[leaving]) Δb: $(Δb[leaving]) c_hat: $(c_hat) perturbation_c_hat: $(perturbation_c_hat) Δc: $(Δc) basic: $(is_basic)"
                @debug "$(pc)"
                throw(ErrorException("Infeasible/Unbounded (minJ)"))
            end

            get_column!(Δb, A, j)
            ftran!(pfi, Δb)
        else
            primal_count += 1
            #primal simplex step
            get_column!(Δb, A, j)
            ftran!(pfi, Δb)
            
            fill!(pb, +Inf64)
            @inbounds for i in 1:length(b_hat)
                if Δb[i] > eps 
                    # the value of the variable will decrease; down to its lower bound (plus perturbation) at maximum.
                    pb[i] = (b_hat[i] + t * perturbation_b_hat[i]) / Δb[i]
                end
            end
            (minL, leaving) = min_argmin(pb)
            if isinf(minL)
                @debug "$iter t_b: $t_b t_c: $t_c entering: $j leaving:? b_hat: $(b_hat) perturbation_b_hat: $(perturbation_b_hat) Δb: $(Δb) c_hat: $(c_hat) perturbation_c_hat: $(perturbation_c_hat) Δc: $(Δc) basic: $(is_basic)"
                @debug "$(pb)"
                throw(ErrorException("Infeasible/Unbounded (minL)"))
            end

            #computeΔc!(Δc, A, el, pfi, is_basic, leaving)
            computeΔc2!(Δc, tA, el, pfi, is_basic, leaving, eps)
            Δz = -Δc
        end
        catch
            #forced_refactoring = true
            rethrow()
        end

        @debug "$iter t_b: $t_b t_c: $t_c entering:$j leaving:$(basis[leaving]) b_hat: $(b_hat[leaving]) perturbation_b_hat: $(perturbation_b_hat[leaving]) Δb: $(Δb[leaving]) c_hat: $(c_hat[j]) perturbation_c_hat: $(perturbation_c_hat[j]) Δc: $(Δc[j])"

        if Δb[leaving] == 0
            perturbation_c = vcat(repeat([1.0], length(c)-length(b)) .+ 5 .* rand(Float64, length(c)-length(b)), repeat([0.0], length(b))) / 100.0
            perturbation_c_hat = copy(perturbation_c)
            perturbation_b = repeat([1.0], length(b)) .+ 5 .* rand(Float64, length(b))
            perturbation_b_hat = copy(perturbation_b)
            ftran!(pfi, perturbation_b_hat)
        
            multipliers = c[basis]
            perturbation_multipliers = perturbation_c[basis]
            btran!(pfi, multipliers)
            btran!(pfi, perturbation_multipliers)
            for i in 1:length(c_hat)
                c_hat[i] = c[i] - dot(A, i, multipliers)
                @assert !isnan(c_hat[i])
                perturbation_c_hat[i] = perturbation_c[i] - dot(A, i, perturbation_multipliers)
                @assert !isnan(perturbation_c_hat[i])
                if abs(perturbation_c_hat[i]) < eps
                    perturbation_c_hat[i] = 0
                end
            end
            continue
        end
        if !forced_refactoring
            #update basis
            sΔb = sparsevec(Δb)
            η = ETAMatrix(leaving, sΔb)
            push!(pfi.eta_matrices, η)
            
            t = b_hat[leaving] / Δb[leaving]
            @assert !isnan(t) & !isinf(t) "t_b: $t_b t_c: $t_c b_hat: $(b_hat[leaving]) perturbation_b_hat: $(perturbation_b_hat[leaving]) Δb: $(Δb[leaving]) c_hat: $(c_hat[j]) perturbation_c_hat: $(perturbation_c_hat[j]) Δc: $(Δc[j]) leaving: $(basis[leaving]) entering: $j"
            perturbation_t = perturbation_b_hat[leaving] / Δb[leaving]
            @assert !isnan(perturbation_t) & !isinf(perturbation_t) "perturbation_b_hat: $(perturbation_b_hat[leaving]) Δb: $(Δb[leaving])"

            axpy!(-t, Δb, b_hat)
            axpy!(-perturbation_t, Δb, perturbation_b_hat)
            b_hat[leaving] = t
            perturbation_b_hat[leaving] = perturbation_t 
                    
            s = c_hat[j] / Δc[j]
            @assert !isnan(s) & !isinf(s) "c_hat: $(c_hat[j]) Δc: $(Δc[j]) b_hat: $(b_hat[leaving]) Δb: $(Δb[leaving]) leaving: $(basis[leaving]) entering: $j"
            perturbation_s = perturbation_c_hat[j] / Δc[j]
            @assert !isnan(perturbation_s) & !isinf(perturbation_s) "perturbation_c_hat: $(perturbation_c_hat[j]) Δc: $(Δc[j])"
            axpy!(-s, Δc, c_hat)
            axpy!(-perturbation_s, Δc, perturbation_c_hat)
            c_hat[basis[leaving]] = -s
            perturbation_c_hat[basis[leaving]] = -perturbation_s            

            # @inbounds for i in 1:length(b_hat)
            #     if abs(b_hat[i]) < eps
            #         b_hat[i] = 0
            #     end
            #     if abs(perturbation_b_hat[i]) < eps
            #         perturbation_b_hat[i] = 0
            #     end
            # end

            # @inbounds for i in 1:length(c_hat)
            #     if abs(c_hat[i]) < eps
            #         c_hat[i] = 0
            #     end
            #     if abs(perturbation_c_hat[i]) < eps
            #         perturbation_c_hat[i] = 0
            #     end
            # end

            is_basic[basis[leaving]] = false
            is_basic[j] = true
            basis[leaving] = j

            for i in basis
                #@assert abs(c_hat[i]) < 1e-3 "$i $(abs(c_hat[i]))"
                c_hat[i] = 0
                perturbation_c_hat[i] = 0
            end
        end

        if iter % 70 == 0 || forced_refactoring
            forced_refactoring = false

            pfi, basis = LUelimination(A, basis)
            
            b_hat = copy(b)
            ftran!(pfi, b_hat)
            @assert count(x -> isnan(x), b_hat) == 0

            perturbation_b_hat = copy(perturbation_b)
            ftran!(pfi, perturbation_b_hat)
            
            multipliers = c[basis]
            perturbation_multipliers = perturbation_c[basis]
            btran!(pfi, multipliers)
            btran!(pfi, perturbation_multipliers)
            for i in 1:length(c_hat)
                c_hat[i] = c[i] - dot(A, i, multipliers)
                @assert !isnan(c_hat[i])
                perturbation_c_hat[i] = perturbation_c[i] - dot(A, i, perturbation_multipliers)
                @assert !isnan(perturbation_c_hat[i])
                if abs(perturbation_c_hat[i]) < eps
                    perturbation_c_hat[i] = 0
                end
            end

            for i in basis
                #@assert abs(c_hat[i]) < 1e-12 "$(abs(c_hat[i]))"
            end
        end
    end
end

end # module
