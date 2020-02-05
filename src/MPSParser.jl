using SparseArrays

import Base.resize!

mutable struct Problem
    c::Array{Float64,1}
    b::Array{Float64,1}
    A::SparseMatrixCSC
    lower_bounds::Array{Float64,1}
    upper_bounds::Array{Float64,1}
    c_names::Array{String}
end

# The following template is a guide for the use of MPS format:

#     ---------------------------------------------------------------------
#     Field:    1           2          3         4         5         6
#     Columns:  2-3        5-12      15-22     25-36     40-47     50-61
#               NAME   problem name
#               ROWS
#                type     name
#               COLUMNS
#                        column       row       value     row      value
#                         name        name                name
#               RHS
#                         rhs         row       value     row      value
#                         name        name                name
#               RANGES
#                         range       row       value     row      value
#                         name        name                name
#               BOUNDS
#                type     bound       column    value
#                         name        name
#               SOS
#                type     CaseName    SOSName   SOSpriority
#                         CaseName    VarName1  VarWeight1
#                         CaseName    VarName2  VarWeight2
#                         CaseName    VarNameN  VarWeightN
#               ENDATA

function get_slack_variables_count(rows, columns)
    count = 0
    for (row_name, v) in sort(collect(rows), by=r->r[2][1])
        if v[2] == "G" || v[2] == "L"
            count = count + 1
        elseif v[2] == "E"
            count = count + 2
        end
    end
    return count
end

function createA(rows, columns, vs)::SparseMatrixCSC{Float64, Int64}
    vs = filter(v -> rows[v[2]][2] != "N", vs)
    
    column_count = maximum(values(columns))
    row_count = length(collect(filter(r -> r[2][2] == "L" || r[2][2] == "G", rows))) + 
                2 * length(collect(filter(r -> r[2][2] == "E", rows)))

    I = Int64[]
    J = Int64[]
    V = Float64[]

    for (column_name, row_name, value) in vs
        type = rows[row_name][2]
        row_index = rows[row_name][1]
        column_index = columns[column_name]
        if type == "L" 
            push!(I, row_index)
            push!(J, column_index)
            push!(V, value)
        elseif type == "G" 
            push!(I, row_index)
            push!(J, column_index)
            push!(V, -value)
        elseif type == "E" 
            push!(I, row_index)
            push!(J, column_index)
            push!(V, value)
            push!(I, row_index+1)
            push!(J, column_index)
            push!(V, -value)
        end
    end

    A = SparseArrays.sparse(I, J, V, row_count, column_count)
    return A
end

function createB(rows, columns, rhs)::Array{Float64}
    row_count = length(collect(filter(r -> r[2][2] == "L" || r[2][2] == "G", rows))) + 
                2 * length(collect(filter(r -> r[2][2] == "E", rows)))

    b = zeros(row_count)
    
    if length(rhs) > 0
        rhs = first(values(rhs))
    else 
        rhs = []
    end
    for (row_name, bound) in rhs
        type = rows[row_name][2]
        row_index = rows[row_name][1]
        if type == "L"
            b[row_index] = bound
        elseif type == "G" 
            b[row_index] = -bound
        elseif type == "E" 
            b[row_index] = bound
            b[row_index+1] = -bound
        end
    end

    return b
end

function createC(rows, columns, vs)::Array{Float64}
    objective_row = first(sort(collect(filter(r -> r[2][2] == "N", rows)), by=r->r[2][1]))

    c = zeros(length(columns))

    for v in filter(v -> v[2] == objective_row[1], vs)
        c[columns[v[1]]] = v[3]
    end

    return c
end

function create_lower_bounds(columns, bounds)::Array{Float64}
    if length(bounds) > 0
        bounds = first(values(bounds))
    else
        bounds = []
    end

    lower_bounds = zeros(length(columns))
    for (column_name, type, value) in bounds
        if type == "LO" || type == "FX"
            lower_bounds[columns[column_name]] = value
        end
        if type == "FR" || type == "MI"
            lower_bounds[columns[column_name]] = -Inf64
        end
    end
    return lower_bounds
end

function create_upper_bounds(columns, bounds)::Array{Float64}
    if length(bounds) > 0
        bounds = first(values(bounds))
    else
        bounds = []
    end

    upper_bounds = repeat([Inf64], length(columns))
    for (column_name, type, value) in bounds
        if type == "UP" || type == "FX"
            upper_bounds[columns[column_name]] = value
        end
        if type == "FR"
            upper_bounds[columns[column_name]] = Inf64
        end
        if type == "MI"
            upper_bounds[columns[column_name]] = 0.0
        end
    end
    return upper_bounds
end

function parseMPS(file_name)
    name = ""
    rows = Dict{String, Tuple{Int64, String}}() # row_name => (index, type)
    columns = Dict{String, Int64}() # column_name => index
    vs = Tuple{String, String, Float64}[] # column_name, row_name, value
    rhs = Dict{String, Dict{String, Float64}}() # name => row_name, value
    bounds = Dict{String, Array{Tuple{String, String, Float64}, 1}}() # name => [(column_name, type, value)]

    firstNrow = true
    excludeNRows = Set{String}()
    row_index = 1
    open(file_name) do f
        section = :NAME
        for l in eachline(f)
            line = rstrip(l)
            if length(lstrip(line)) == 0 || line[1] == '*'
                continue
            end
            if section == :NAME && startswith(line, "NAME")
                name = line[15:end]
                continue
            end
            if section == :NAME && startswith(line, "ROWS")
                section = :ROWS
                continue
            end
            if section == :ROWS && startswith(line, "COLUMNS")
                section = :COLUMNS
                continue
            end
            if section == :COLUMNS && startswith(line, "RHS")
                section = :RHS
                continue
            end
            if startswith(line, "BOUNDS")
                section = :BOUNDS
                continue
            end
            if startswith(line, "RANGES")
                section = :RANGES
                continue
            end
            if startswith(line, "ENDATA")
                section = :END
                continue
            end
            
            if section == :RANGES
                throw(ErrorException("RANGES not supported"))
            end
            
            if section == :ROWS
                type = String(strip(line[2:3]))
                row_name = String(strip(line[4:end]))
                if type == "N" 
                    if firstNrow
                        push!(rows, row_name => (0, type))
                        firstNrow = false
                    else
                        push!(excludeNRows, row_name)
                    end
                else
                    push!(rows, row_name => (row_index, type))
                    if type == "G" || type == "L"
                        row_index = row_index + 1
                    else 
                        @assert type == "E" "$type == E"
                        row_index = row_index + 2
                    end
                end
                continue
            end
            if section == :COLUMNS
                try
                    #5-12      15-22     25-36     40-47     50-61
                    column_name = String(strip(line[5:12]))
                    if !haskey(columns, column_name)
                        push!(columns, column_name => length(columns)+1)
                    end
    
                    row_name = String(strip(line[15:22]))
                    if !(row_name in excludeNRows)
                        value = parse(Float64, String(strip(line[25:min(36,length(line))])))
                        push!(vs, (column_name, row_name, value))
                    end

                    if length(line) > 40
                        row_name2 = String(strip(line[40:47]))
                        if !(row_name2 in excludeNRows)
                            if !isempty(row_name2)
                                value2 = parse(Float64, String(strip(line[50:end])))
                                push!(vs, (column_name, row_name2, value2))
                            end
                        end
                    end
                    continue
                catch e
                    println("$line $e")
                    rethrow(e)
                end
            end
            if section == :RHS
                rhs_name = String(strip(line[5:12]))
                row_name = String(strip(line[15:22]))
                if !(row_name in excludeNRows)
                    if !haskey(rhs, rhs_name)
                        push!(rhs, rhs_name => Dict{String, Float64}())
                    end
                    value = parse(Float64, String(strip(line[25:min(36,length(line))])))
                    push!(rhs[rhs_name], row_name => value)
                end

                if length(line) > 40
                    row_name2 = String(strip(line[40:47]))
                    if !(row_name2 in excludeNRows)
                        if !isempty(row_name2)
                            value2 = parse(Float64, String(strip(line[50:end])))
                            push!(rhs[rhs_name], row_name2 => value2)
                        end
                    end
                end
                continue
            end
            if section == :BOUNDS
                try
                    type = String(strip(line[2:3]))
                    name = String(strip(line[5:12]))
                    column = String(strip(line[15:min(22,length(line))]))
                    if type == "PL"
                        continue    
                    elseif type in ["FR", "MI"]
                        value = 0.0
                    elseif type in ["UP","LO", "FX"]
                        value = parse(Float64, String(strip(line[25:min(36,length(line))])))
                        # if type == "LO" && value < 0
                        #     throw(ErrorException("Does not support negative lower bounds: $line"))
                        # end
                    else
                        throw(ErrorException("Cannot process bound: $line"))
                    end
                    if !haskey(bounds, name)
                        push!(bounds, name => Tuple{String, String, Float64}[])
                    end
                    push!(bounds[name], (column, type, value))
                catch e
                    println("$line $e")
                    rethrow(e)
                end
            end
        end
    end

    A = createA(rows, columns, vs)
    c = createC(rows, columns, vs)
    b = createB(rows, columns, rhs)
    lo = create_lower_bounds(columns, bounds)
    up = create_upper_bounds(columns, bounds)
    names = ["" for i in 1:maximum(values(columns))]
    for (name, index) in columns
        names[index] = name
    end
    return Problem(c, b, A, lo, up, names)
end