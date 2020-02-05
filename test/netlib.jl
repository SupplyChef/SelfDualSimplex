using Test
using SelfDualSimplex

function run(name::String, value)
    print("$name ")
    p = parseMPS("C:\\Users\\rlecoeuc\\.julia\\dev\\SelfDualSimplex\\Benchmarks\\netlib\\$(name).SIF")
    return run(p, value)    
end

@testset "Netlib" begin
    @test begin
        run("AFIRO", -0.46475314285714285714285714285714e3)
        true
    end

    @test begin
        run("ADLITTLE", 0.22549496316238038228101176621492e6)
        true
    end

    @test begin
        run("AGG", -0.35991767286576506712640824319636e8)
        run("AGG2", -0.20239252355977109024317661926133e8)
        run("AGG3", 0.10312115935089225579061058796215e8)
        true
    end

    @test begin
        run("BANDM", -1.5862801845e2)
        true
    end

    @test begin
        run("BEACONFD", 3.3592485807e4)
        true
    end

    @test begin
        run("BLEND", -0.30812149845828220173774356124984e2)
        true
    end

    @test begin
        run("BNL1", 0.19776295615228892439564398331821e4)
        run("BNL2", 0.1811236540358545170448413697691e4)
        true
    end

    # @test begin
    #     run("BOEING1", -3.3521356751e2)
    #     true
    # end

    @test begin
        run("BORE3D", 1.3730803942e3)
        true
    end

    @test begin
        run("BRANDY", 1.5185098965e3)
        true
    end    

    @test begin
        run("CAPRI", 2.6900129138e3)
        true
    end    

    @test begin
        run("CYCLE", -5.2263930249e0)
        true
    end    

    @test begin
        run("CZPROB", 0.21851966988565774858951155947191e7)
        true
    end    

    @test begin
        run("DEGEN2", -0.1435178e4)
        run("DEGEN3", -9.8729400000e2)
        true
    end

    @test begin
        run("D6CUBE", 3.1549166667e2)
        true
    end

    @test begin
        run("DFL001", 0.11266396046671392202377652175477e8)
        true
    end

    @test begin
        run("E226", -0.18751929066370549102605687681285e2)
        true
    end

    @test begin
        run("ETAMACRO", -0.7557152333749133350792583667773e3)
        true
    end

    @test begin
        run("FFFFF800", 0.555679564817496376532864378969e6)
        true
    end

    @test begin
        run("FINNIS", 0.17279106559561159432297900375543e6)
        true
    end

    @test begin
        run("FIT1D", -0.91463780924209269467749025024617e4)
        run("FIT1P", 0.91463780924209269467749025024617e4)
        run("FIT2D", -0.68464293293832069575943518435837e5)
        run("FIT2P", 0.68464293293832069575943518435837e5)
        true
    end

    @test begin
        run("FORPLAN", -0.66421896127220457481235119701692e3)
        true
    end

    @test begin
        run("GFRD-PNC", 0.69022359995488088295415596232193e7)
        true
    end

    @test begin
        run("GROW7", -0.47787811814711502616766956242865e8)
        run("GROW15", -0.10687094129357533671604040930313e9)
        run("GROW22", -0.16083433648256296718456039982613e9)
        true
    end

    @test begin
        run("ISRAEL", -0.89664482186304572966200464196045e6)
        true
    end

    @test begin
        run("KB2", -0.17499001299062057129526866493726e4)
        true
    end    

    @test begin
        run("LOTFI", -0.2526470606188e2)
        true
    end

    @test begin
        run("PILOT", -0.55748972928406818073034256636894e3)
        true
    end

    @test begin
        run("PILOT4", -0.25811392588838886745830997266797e4)
        true
    end

    @test begin
        run("PILOTNOV", -0.44972761882188711430996211783943e4)
        true
    end

    @test begin
        run("PILOT-JA", -0.61131364655813432748848620538024e4)
        true
    end

    @test begin
        run("PILOT-WE", -0.27201075328449639629439185412556e7)
        true
    end

    # @test begin
    #     run("NESM", 0.14076036487562728337980641375835e8)
    #     true
    # end

    @test begin
        run("RECIPELP", -2.6661600000e2)
        true
    end    

    @test begin
        run("SC50A", -0.64575077058564509026860413914575e2)
        run("SC50B", -0.7e2)
        true
    end

    @test begin
        run("SCAGR7", -0.2331389824330984e7)
        run("SCAGR25", -1.4753433061e7)
        true
    end

    @test begin
        run("SCTAP1", 1.4122500000e3)
        run("SCTAP2", 1.7248071429e3)
        run("SCTAP3", 1.4240000000e3)
        true
    end

    @test begin
        run("SCFXM1", 1.8416759028e4)
        run("SCFXM2", 3.6660261565e4)
        run("SCFXM3", 5.4901254550e4)
        true
    end

    @test begin
        run("SHARE1B", -7.6589318579e4)
        run("SHARE2B", -0.41573224074141948654519910873841e3)
        true
    end

    @test begin
        run("SHELL", 1.2088253460e9)
        true 
    end    

    @test begin
        run("SIERRA", 0.1539436218363193e8)
        true
    end

    @test begin
        run("STAIR", -0.25126695119296330352803637106304e3)
        true
    end

    @test begin
        run("STANDATA", 0.12576995e4)
        true
    end

    @test begin
        run("STANDMPS", 0.14060175e4)
        true
    end
    
    @test begin
        run("SCSD8", 9.0499999993e2)
        true
    end

    @test begin
        run("80BAU3B", 9.8723216072e5)
        true
    end

    @test begin
        run("GREENBEA", -72555248.13)
        run("GREENBEB", -0.43022602612065867539213672544432e7)
        true
    end

    @test begin
        run("SCORPION", 0.18781248227381066296479411763586e4)
        true
    end

    @test begin
        run("SHIP04L", 0.17933245379703557625562556255626e7)
        run("SHIP08S", 0.1920098210534619710172695474693e7)
        run("SHIP12L", 1.4701879193e6)
        run("SHIP12S", 1.4892361344e6)
        true
    end 

    @test begin
        run("STOCFOR1", -0.41131976219436406065682760731514e5)
        run("STOCFOR2", -0.39024408537882029604587908772433e5)
        run("STOCFOR3", -0.39976783943649587403509204700686e5)
        true
    end

    @test begin
        run("TRUSS", 4.5881584719e5)
        true
    end

    @test begin
        run("TUFF", 2.9214776509e-1)
        true
    end

    @test begin
        run("VTP-BASE", 1.2983146246e5)
        true
    end

    @test begin
        run("25FV47", 0.55018458882867447945812325883916e4)
        true
    end

    @test begin
        run("CRE-A", 0.23595407060971607914108674680625e8)
        run("CRE-B", 0.23129639886832364609512881638847e8)
        run("CRE-C", 0.25275116140880216542103232084057e8)
        run("CRE-D", 0.2445496976454924819803581786941e8)
        true
    end

    @test begin
        run("D2Q06C", 0.12278421081418945895739128812392e6)
        true
    end

    @test begin
        run("KEN-07", -0.679520443381687e9)
        true
    end

    @test begin
        run("KEN-11", -0.6972382262519971e10)
        true
    end

    @test begin
        run("KEN-13", -0.10257394789482432e11)
        true
    end    

    # @test begin
    #     run("KEN-18", -0.5221702528739681e11)
    #     true
    # end    

    @test begin
        run("MAROS", -0.58063743701125895401208534974734e5)
        true
    end

    @test begin
        run("MAROS-R7", 0.14971851664796437907337543903552e7)
        true
    end

    @test begin
        run("PDS-02", 0.2885786201e11)
        run("PDS-06", 0.277610376e11)        
        run("PDS-10", 0.26727094976e11)
        true
    end

    @test begin
        run("PDS-20", 0.2382165864e11)
        true
    end

    @test begin
        run("PDS-40", 0.18855198824e11)
        true
    end

    @test begin
        run("MODSZK1", 0.32061972906431580494333823530763e3)
        true
    end

    @test begin
        run("WOODW", 0.1304476333084229269005552085566e1)
        true
    end

    @test begin
        run("WOOD1P", 0.14429024115734092400010936668043e1)
        true
    end

    @test begin
        run("OSA-07", 0.53572251729935104299116808137737e6)
        true
    end

    @test begin
        run("OSA-14", 0.11064628447362547969656403391343e7)
        true
    end

    @test begin
        run("OSA-30", 0.21421398732097579473449352967425e7)
        true
    end

    @test begin
        run("OSA-60", 0.40440725031574228960285586791611e7)
        true
    end

    @test begin
        run("QAP8", 2.0350000000e+02)
        true
    end

    @test begin
        run_m("dbic1", -9.768973e6)
        true
    end

    # @test begin
    #     run("QAP12", 5.228943506e2)
    #     true
    # end    

    # @test begin
    #     run_lp("Linf_520c", 0.19886847)
    #     true
    # end

    @test begin
        run_lp("fome13", 90131168.37)
        true
    end
end