using Test
using SelfDualSimplex

function run_m(name::String, value)
    print("$name ")
    p = parseMPS("../benchmarks/meszaros/$(name).mps")
    return run(p, value)    
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
        run_m("air06", 49616.36354)
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
        run_m("nemsemm1", 512959.6008)
        run_m("model2", -7.400489e3)
        #run_m("delfland", 2.203077873)
        #run_m("p19", 253964.3546)
        #run_m("dsbmip", -305.198175)
        run_m("f2177", 90.00)
        true
    end

    @test begin
        run_m("nsic1", -9.168554e6)
        run_m("nsic2", -8.203512e6)
        true
    end

    @test begin
        run_m("dbir1", -8.1067070000e6)
        run_m("dbir2", -6.1169165000e6)
        true
    end

    @test begin
        run_m("radio", 1.0)
        true
    end
end