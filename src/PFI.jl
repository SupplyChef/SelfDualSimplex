include("ETA.jl")
include("LU.jl")

using SparseArrays

import Base.*

struct PFI
    eta_matrices::Array{ETAMatrix, 1}
    luf
    function PFI(eta_matrices::Array{ETAMatrix, 1})
        pfi = new(eta_matrices, nothing)
        return pfi
    end
    function PFI(eta_matrices::Array{ETAMatrix, 1}, luf)
        pfi = new(eta_matrices, luf)
        return pfi
    end
end

function ftran!(pfi::PFI, x::Array{Float64, 1})
    @inbounds for eta in pfi.eta_matrices
        ftran!(eta, x)
    end
    if !isnothing(pfi.luf)
        y = pfi.luf \ x
        for i in 1:length(y)
            x[i] = y[i]
        end
    end
end

function btran!(pfi::PFI, x::Array{Float64, 1})
    if !isnothing(pfi.luf)
        y = pfi.luf' \ x
        for i in 1:length(y)
            x[i] = y[i]
        end
    end
    @inbounds for i in length(pfi.eta_matrices):-1:1
        btran!(pfi.eta_matrices[i], x)
    end
end