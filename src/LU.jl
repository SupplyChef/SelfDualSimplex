using SparseArrays
using LinearAlgebra
using Printf

include("PFI.jl")

function LUdecomposition(m::SparseMatrixCSC{Float64, Int64}, basis)::PFI
    luf  = lu(m[:,basis])
    return PFI(ETAMatrix[], luf, basis)
end

function LUelimination!(pfi::PFI, m::SparseMatrixCSC{Float64, Int64}, basis::Array{Int64, 1})
    tmp = [get_column(m, b) for b in basis]
    map_to_new_indexes = collect(1:length(basis))

    tmp_row_nzs = [Array{Int64,1}() for row in 1:length(basis)]
    for j in 1:length(basis)
        for i in 1:length(tmp[j].nzind)
            if tmp[j].nzval[i] != 0.0
                push!(tmp_row_nzs[tmp[j].nzind[i]], j)
            end
        end
    end
    
    ls = ETAMatrix[]
    us = ETAMatrix[]

    col_nnzs = zeros(Int64, length(basis))
    @inbounds for t in 1:length(basis)
        col_nnzs[t] = get_nnz(tmp[t])
    end

    row_j = zeros(length(basis))

    Ib = zeros(Int64, length(basis))
    Vb = zeros(Float64, length(basis))
    @inbounds for j in 1:length(basis)
        # println(j)
        # A = zeros(Float64, (length(tmp),length(tmp)))
        # for u in 1:length(tmp)
        #     A[:, u] = tmp[u]
        # end
        # for v in 1:length(tmp)
        #     println(A[v, :])
        # end
        # println(tmp_row_nzs)

        max_pivot = 0.0
        @inbounds for tmp_j in tmp_row_nzs[j]
            r = tmp[tmp_j][j]
            if r > max_pivot || -r > max_pivot
                max_pivot = abs(r)
            end
            row_j[tmp_j] = r
        end

        if max_pivot == 0.0
            throw(ErrorException("Singular basis"))# $j $basis"))
        end

        min_nnz = Inf64
        tmp_k = -1
        @inbounds for tmp_j in tmp_row_nzs[j]
            if abs(row_j[tmp_j]) >= 0.01 * max_pivot #|| abs(row_j[tmp_j]) >= 0.1
                if col_nnzs[tmp_j] < min_nnz
                    min_nnz = col_nnzs[tmp_j]
                    tmp_k = tmp_j
                end
            end
        end
        
        map_to_new_indexes[j] = tmp_k
        pivot = row_j[tmp_k]
                
        lfactors = spzeros(length(basis))
        ufactors = spzeros(length(basis))
        @inbounds for k in 1:length(tmp[tmp_k].nzind)
            v = tmp[tmp_k].nzval[k]
            if abs(v) < 1e-15
                continue
            end
            i = tmp[tmp_k].nzind[k]
            if i < j
                ufactors[i] = v
            elseif i > j
                lfactors[i] = v / pivot
            end
        end

        if length(lfactors.nzind) > 0
            push!(ls, ETAMatrix(j, 1.0, lfactors))
        end
        if pivot != 1.0 || length(ufactors.nzind) > 0
            push!(us, ETAMatrix(j, pivot, ufactors))
        end
                
        @inbounds for l in tmp_row_nzs[j]
            if l == tmp_k
                continue
            end
            t = row_j[l] / pivot 
            if t != 0.0
                tmp[l] = eliminate2(tmp[tmp_k], tmp[l], j, t, tmp_row_nzs, l)                
            end
        end
        t = row_j[tmp_k] / pivot 
        tmp[tmp_k] = eliminate2(tmp[tmp_k], tmp[tmp_k], j, t, tmp_row_nzs, tmp_k)

        for tmp_j in tmp_row_nzs[j]
            row_j[tmp_j] = 0.0
        end
    end
    basis = basis[map_to_new_indexes]

    @debug "LU factorization: $(length(ls)) lower Eta matrices, $(length(us)) upper Eta matrices."
    pfi.eta_matrices = vcat(ls,reverse(us))
    pfi.basis = basis
end

function push_if_not_present!(a::Array{Int64, 1}, new::Int64)
    found = false
    for old in a
        if old == new
            found = true
            break
        end
    end
    if !found 
        push!(a, new)
    end
end

function remove_if_present!(a, new)
    filter!(e -> e ≠ new, a)
end

function eliminate2(a::SparseVector{Float64, Int64}, b::SparseVector{Float64, Int64}, j::Int64, f::Float64, nzs, l)::SparseVector{Float64, Int64}
    @inbounds for ia in 1:length(a.nzind)
        i = a.nzind[ia] 
        if i > j
            v = a.nzval[ia]
            old_v = b[i]
            new_v = old_v - f * v
            if abs(new_v) < 1e-15
                b[i] = 0.0
                if old_v != 0.0
                    remove_if_present!(nzs[i], l)
                end
            else
                b[i] = new_v 
                if old_v == 0.0
                    push_if_not_present!(nzs[i], l)
                end
            end
        end
    end
    return b
end