module SelfDualSimplex

import  Base.*
import  Base.size
import  Base.getindex
import  Base.setindex

using Dates
using DelimitedFiles
using LinearAlgebra
using SparseArrays
using SuiteSparse

include("MPSParser.jl")
include("LU.jl")
include("Presolve.jl")

export PFI
export ETAMatrix
export ftran!
export btran!
export solve

export parseMPS
export Problem
export add_upper_bounds!
export add_lower_bounds!
export handle_negative_lowerbound_variables!
export add_slack_variables!

export LUelimination!
export LUdecomposition

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
            # (Is, Vs) = get_nz(tA, i)
            # for j in 1:length(Is)
            #     Δc[Is[j]] += el[i] * Vs[j]
            # end
            @inbounds for j in tA.colptr[i]:(tA.colptr[i+1]-1)
                Δc[tA.rowval[j]] += el[i] * tA.nzval[j]
            end
        end
    end

    for i in (c - b + 1):length(Δc)
        if i > c - b
            Δc[i] = el[i - (c - b)]
        end
        #else 
        #if is_basic[i]
        #     Δc[i] = 0
        # end
        # if abs(Δc[i]) < 1e-12
        #     Δc[i] = 0
        # end
        #end
    end
end

function solve(p; time_limit=-1)
    p = deepcopy(p)
    print("$(size(p.A)) ")
    presolution = Dict{String, Float64}()
    presolve!(p, presolution)
    print("$(size(p.A)) ")
    handle_negative_lowerbound_variables!(p)
    add_upper_bounds!(p)
    add_lower_bounds!(p)
    #add_free_variables!(p)    
    add_slack_variables!(p)
    print("$(size(p.A)) ")
    solution = solve(p.A, p.c, p.b; time_limit=time_limit)
    solution2 = Dict{String, Float64}(p.c_names[i] => get(solution, i, 0.0) for i in 1:length(p.c_names))
    merge!(solution2, presolution)
    return solution2
end

function solve(A::SparseMatrixCSC{Float64, Int64},c::Array{Float64,1},b::Array{Float64,1}; time_limit=-1)
    SuiteSparse.UMFPACK.umf_ctrl[8] = 0

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
    pfi = PFI()
    #LUelimination!(pfi, A, basis)
    pfi = LUdecomposition(A, basis)
    basis = pfi.basis

 
    b_hat = copy(b) # values of basic variables
    c_hat = copy(c) # values of dual variables

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

    pb = zeros(length(b))
    pc = zeros(length(c))

    forced_refactoring = false
    primal_count = 0
    dual_count = 0
    start = Dates.now()
    while(time_limit < 0 || (Dates.now() - start < Second(time_limit)))
        #@assert sum(is_basic) == length(basis)
        iter = iter + 1

        fill!(pb, 0.0)
        @inbounds for i in 1:length(b_hat)
            if b_hat[i] < 0.0
                violation = -b_hat[i]
                pb[i] = violation / perturbation_b_hat[i]
            end
        end
        fill!(pc, 0.0)
        @inbounds for i in 1:length(c_hat)
            c_hat_i = c_hat[i]
            if c_hat_i < 0.0 && !is_basic[i] 
                violation = -c_hat_i 
                pc[i] = violation / perturbation_c_hat[i]
            end
        end

        (t_b, leaving) = max_argmax(pb)
        (t_c, j) = max_argmax(pc)
        t = max(t_b, t_c)
        
        #@assert t <= old_t "$t $old_t"
        old_t = t

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
                
            fill!(pc, Inf64)
            @inbounds for i in 1:length(c_hat)
                if !is_basic[i] && Δc[i] < -eps  #c will decrease; it should not go negative for variables that are not at their upper bound (or we loose optimality)
                    #@assert (c_hat[i] + t * perturbation_c_hat[i]) > 0 "iter: $iter c_hat: $(c_hat[i]) perturbation_c_hat: $(perturbation_c_hat[i]) t: $t $(c_hat[i] + t * perturbation_c_hat[i]) > 0"
                    pc[i] = (c_hat[i] + t * perturbation_c_hat[i]) / -Δc[i]
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
        end
        catch
            #forced_refactoring = true
            rethrow()
        end

        @debug "$iter t_b: $t_b t_c: $t_c entering:$j leaving:$(basis[leaving]) b_hat: $(b_hat[leaving]) perturbation_b_hat: $(perturbation_b_hat[leaving]) Δb: $(Δb[leaving]) c_hat: $(c_hat[j]) perturbation_c_hat: $(perturbation_c_hat[j]) Δc: $(Δc[j])"

        if Δb[leaving] == 0 || Δc[j] == 0
            println("$iter Changing perturbation")
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
            
            updateBasicVariables(b_hat, Δb, leaving)
            updateBasicVariables(perturbation_b_hat, Δb, leaving)
            
            updateDualVariables(c_hat, Δc, j, leaving, basis)
            updateDualVariables(perturbation_c_hat, Δc, j, leaving, basis)
            
            # @inbounds for i in 1:length(b_hat)
            #     if b_hat[i] < 0
            #         b_hat[i] = 0
            #     end
            #     if perturbation_b_hat[i] < 0
            #         perturbation_b_hat[i] = 0
            #     end
            # end

            # @inbounds for i in 1:length(c_hat)
            #     if c_hat[i] < 0
            #         c_hat[i] = 0
            #     end
            #     if perturbation_c_hat[i] < 0
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

            #LUelimination!(pfi, A, basis)
            pfi = LUdecomposition(A, basis)
            basis = pfi.basis
            
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

function updateBasicVariables(b_hat, Δb, leaving)
    @inbounds t = b_hat[leaving] / Δb[leaving]
    #@assert !isnan(t) & !isinf(t) "t_b: $t_b t_c: $t_c b_hat: $(b_hat[leaving]) perturbation_b_hat: $(perturbation_b_hat[leaving]) Δb: $(Δb[leaving]) c_hat: $(c_hat[j]) perturbation_c_hat: $(perturbation_c_hat[j]) Δc: $(Δc[j]) leaving: $(basis[leaving]) entering: $j"
    axpy!(-t, Δb, b_hat)
    @inbounds b_hat[leaving] = t
end

function updateDualVariables(c_hat, Δc, j, leaving, basis)
    @inbounds s = c_hat[j] / Δc[j]
    #@assert !isnan(s) & !isinf(s) "t_b:$t_b t_c:$t_c c_hat: $(c_hat[j]) Δc: $(Δc[j]) b_hat: $(b_hat[leaving]) Δb: $(Δb[leaving]) leaving: $(basis[leaving]) entering: $j"
    axpy!(-s, Δc, c_hat)
    @inbounds c_hat[basis[leaving]] = -s
end

end # module
