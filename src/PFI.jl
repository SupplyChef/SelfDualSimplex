include("ETA.jl")

using SparseArrays
using LinearAlgebra
using SuiteSparse.UMFPACK

import Base.*

mutable struct PFI
    eta_matrices::Array{ETAMatrix, 1}
    luf::Union{UmfpackLU{Float64,Int64}, Nothing}
    basis::Array{Int64, 1}

    function PFI()
        pfi = new(ETAMatrix[], nothing, Int64[])
        return pfi
    end
    function PFI(basis::Array{Int64, 1}, eta_matrices::Array{ETAMatrix, 1})
        pfi = new(eta_matrices, nothing, basis)
        return pfi
    end
    function PFI(eta_matrices::Array{ETAMatrix, 1}, luf)
        pfi = new(eta_matrices, luf, Int64[])
        return pfi
    end
    function PFI(eta_matrices::Array{ETAMatrix, 1}, luf, basis::Array{Int64, 1})
        pfi = new(eta_matrices, luf, basis)
        return pfi
    end
end

function ftran!(pfi::PFI, x::Array{Float64, 1})
    if !isnothing(pfi.luf)
        ldiv!(pfi.luf, x)
    end
    @inbounds for eta in pfi.eta_matrices
        ftran!(eta, x)
    end
end

function btran!(pfi::PFI, x::Array{Float64, 1})
    @inbounds for i in length(pfi.eta_matrices):-1:1
        btran!(pfi.eta_matrices[i], x)
    end
    if !isnothing(pfi.luf)
        ldiv!(pfi.luf', x)
    end
end