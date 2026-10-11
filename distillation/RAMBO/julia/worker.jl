include("model.jl")
PowerModels.silence()
BLAS.set_num_threads(1)
for line in eachline(stdin)
    try
        request = JSON.parse(line)
        if request["stage"] == "versions"
            result = Dict("julia"=>string(VERSION), "JuMP"=>string(pkgversion(JuMP)),
                          "Ipopt"=>string(pkgversion(Ipopt)), "PowerModels"=>string(pkgversion(PowerModels)))
        else
            t = time()
            stage = request["stage"]
            result = stage == "opf" ? final_opf(request) : stage == "warm_opf" ? warm_opf(request) : optimized(request)
            result["wall_seconds"] = time() - t
        end
        println("RAMBO_RESULT ", JSON.json(result))
    catch error
        result = Dict("success"=>false, "status"=>"ERROR", "error"=>sprint(showerror, error, catch_backtrace()))
        println("RAMBO_RESULT ", JSON.json(result))
    end
    flush(stdout)
end
