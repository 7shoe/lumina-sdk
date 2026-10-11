# RAMBO's relaxed KKT problem, derived from a single polar AC-OPF formulation.
# Branch flows are exact expressions (eliminated variables); fixed controls are
# constants. Only lower-level state variables receive stationarity conditions.
using JuMP, Ipopt, PowerModels, JSON, Random, LinearAlgebra
const MOI = JuMP.MOI

number(x::Number) = Float64(x)
number(x) = value(x)
grad(m, x::Number) = Dict{VariableRef,Any}()
grad(m, x) = JuMP.gradient(m, x)

function optimizer(seconds)
    return optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0,
        "sb" => "yes", "max_cpu_time" => Float64(seconds), "tol" => 1e-8,
        "acceptable_tol" => 1e-7, "bound_relax_factor" => 0.0,
        "max_iter" => 2000)
end

function primal(net, seconds; start=nothing, rng=MersenneTwister(42))
    m = Model(optimizer(seconds))
    h, g, href, gref = Any[], Any[], Any[], Any[]
    lower = VariableRef[]
    function box(name, lows, highs; is_lower=true)
        out = Any[]
        for i in eachindex(lows)
            lo, hi = lows[i], highs[i]
            if lo == hi
                push!(out, Float64(lo))
                continue
            end
            initial = start === nothing ? lo + rand(rng) * (hi - lo) : start[name][i]
            x = @variable(m, base_name="$(name)[$i]", start=initial)
            set_lower_bound(x, lo); set_upper_bound(x, hi)
            if is_lower
                push!(lower, x)
                append!(g, [lo - x, x - hi])
                append!(gref, [LowerBoundRef(x), UpperBoundRef(x)])
            end
            push!(out, x)
        end
        return out
    end
    n, ng = net["n"], net["g"]
    theta = Any[]
    for i in 1:n
        if i == net["ref"]
            push!(theta, 0.0)
        else
            initial = start === nothing ? .4 * (rand(rng) - .5) : start["theta"][i]
            x = @variable(m, base_name="theta[$i]", start=initial)
            push!(theta, x); push!(lower, x)
        end
    end
    vm = box("vm", net["vmin"], net["vmax"])
    pg = box("pg", net["pmin"], net["pmax"])
    qg = box("qg", net["qmin"], net["qmax"])
    pd = box("pd", net["pd_lo"], net["pd_hi"]; is_lower=false)
    qd = box("qd", net["qd_lo"], net["qd_hi"]; is_lower=false)
    point = Dict("theta"=>theta, "vm"=>vm, "pg"=>pg, "qg"=>qg, "pd"=>pd, "qd"=>qd)
    pf, qf, pt, qt = Any[], Any[], Any[], Any[]
    for e in eachindex(net["fr"])
        f, t = net["fr"][e], net["to"][e]
        d = theta[f] - theta[t]
        vft = vm[f] * vm[t]
        push!(pf, net["ff_re"][e]*vm[f]^2 + vft*(net["ft_re"][e]*cos(d) + net["ft_im"][e]*sin(d)))
        push!(qf, -net["ff_im"][e]*vm[f]^2 + vft*(net["ft_re"][e]*sin(d) - net["ft_im"][e]*cos(d)))
        push!(pt, net["tt_re"][e]*vm[t]^2 + vft*(net["tf_re"][e]*cos(d) - net["tf_im"][e]*sin(d)))
        push!(qt, -net["tt_im"][e]*vm[t]^2 + vft*(-net["tf_re"][e]*sin(d) - net["tf_im"][e]*cos(d)))
        for expr in (net["angmin"][e] - d, d - net["angmax"][e])
            push!(g, expr); push!(gref, @constraint(m, expr <= 0))
        end
        if net["rate_a"][e] > 0
            for expr in (pf[e]^2 + qf[e]^2 - net["rate_a"][e]^2,
                         pt[e]^2 + qt[e]^2 - net["rate_a"][e]^2)
                push!(g, expr); push!(gref, @constraint(m, expr <= 0))
            end
        end
    end
    for b in 1:n
        gens = findall(==(b), net["gen_bus"])
        loads = findall(==(b), net["load_bus"])
        fr, to = findall(==(b), net["fr"]), findall(==(b), net["to"])
        active = sum(pg[i] for i in gens; init=0.0) - sum(pd[i] for i in loads; init=0.0) -
                 net["gs"][b]*vm[b]^2 - sum(pf[e] for e in fr; init=0.0) - sum(pt[e] for e in to; init=0.0)
        reactive = sum(qg[i] for i in gens; init=0.0) - sum(qd[i] for i in loads; init=0.0) +
                 net["bs"][b]*vm[b]^2 - sum(qf[e] for e in fr; init=0.0) - sum(qt[e] for e in to; init=0.0)
        for expr in (active, reactive)
            push!(h, expr); push!(href, @constraint(m, expr == 0))
        end
    end
    cost = sum(net["c2"][i]*pg[i]^2 + net["c1"][i]*pg[i] + net["c0"][i] for i in 1:ng)
    return (; m, point, lower, h, g, href, gref, cost)
end

function distance(point, history, subsets, delta)
    # Independent history/coordinate subsets per physical block, as upstream.
    total = 0.0
    for key in ("pg", "pd", "vm"), term in subsets[key]
        old = history[term["sample"]][key]
        total += log(delta + sum(abs(point[key][i] - old[i]) for i in term["variables"]))
    end
    return total
end

function kkt!(p, epsilon, warm)
    m = p.m
    lam = @variable(m, [1:length(p.h)], base_name="lambda")
    mu = @variable(m, [1:length(p.g)], lower_bound=0, base_name="mu")
    cost_gradient = grad(m, p.cost)
    stationarity = Dict{VariableRef,Any}(x=>get(cost_gradient, x, 0.0) for x in p.lower)
    for (exprs, multipliers, key) in ((p.h, lam, "lambda"), (p.g, mu, "mu"))
        for (i, expr) in enumerate(exprs)
            set_start_value(multipliers[i], warm[key][i])
            for (x, derivative) in grad(m, expr)
                if haskey(stationarity, x)
                    stationarity[x] += multipliers[i] * derivative
                end
            end
        end
    end
    equations = [JuMP.simplify(m, stationarity[x]) for x in p.lower]
    for expression in equations
        @constraint(m, expression == 0)
    end
    for i in eachindex(mu)
        @constraint(m, mu[i]*p.g[i] >= -epsilon)
    end
    return (; lam, mu, equations)
end

function optimized(request)
    start = get(request, "warm", nothing)
    p = primal(request["network"], request["seconds"]; start=start === nothing ? nothing : start["point"],
               rng=MersenneTwister(request["seed"]))
    opt = request["stage"] == "bilevel" ? kkt!(p, request["epsilon"], start) : nothing
    objective = distance(p.point, request["history"], request["subsets"], request["log_delta"])
    @objective(p.m, Max, objective)
    optimize!(p.m)
    status = string(termination_status(p.m))
    good = termination_status(p.m) in (MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED)
    record = Dict{String,Any}("status"=>status, "success"=>false, "solve_seconds"=>solve_time(p.m))
    if !has_values(p.m)
        return record
    end
    point = Dict{String,Any}(k=>number.(v) for (k,v) in p.point)
    point["cost"] = number(p.cost)
    primal_error = max(maximum(abs.(number.(p.h)); init=0.0), maximum(number.(p.g); init=0.0))
    record["point"] = point
    record["objective"] = objective_value(p.m)
    record["primal_error"] = primal_error
    record["success"] = good && all(isfinite, vcat([v for (k,v) in point if k != "cost"]...)) &&
                         isfinite(point["cost"]) && primal_error <= request["tolerance"]
    if opt === nothing
        record["lambda"] = dual.(p.href) # Feasibility stage maximizes distance.
        record["mu"] = abs.(dual.(p.gref))
    else
        record["lambda"], record["mu"] = value.(opt.lam), value.(opt.mu)
        record["stationarity"] = maximum(abs.(number.(opt.equations)); init=0.0)
        record["complementarity"] = maximum(abs.(value.(opt.mu) .* number.(p.g)); init=0.0)
        record["dual_violation"] = maximum(-value.(opt.mu); init=0.0)
        record["success"] &= record["stationarity"] <= request["tolerance"] &&
            record["dual_violation"] <= request["tolerance"] &&
            record["complementarity"] <= request["epsilon"] + request["tolerance"]
    end
    return record
end

function warm_opf(request)
    # Numerical initialization only: preserve the requested epsilon, objective
    # and feasible region of the subsequent bilevel solve. This additional
    # lower-level solve is explicitly counted, and may be disabled in config.
    initial = request["warm"]["point"]
    p = primal(request["network"], request["seconds"]; start=initial)
    for key in ("pd", "qd"), (i,x) in enumerate(p.point[key])
        if x isa VariableRef
            fix(x, initial[key][i]; force=true)
        end
    end
    @objective(p.m, Min, p.cost)
    optimize!(p.m)
    status = termination_status(p.m)
    result = Dict{String,Any}("status"=>string(status), "success"=>false, "solve_seconds"=>solve_time(p.m))
    if status ∉ (MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED) || !has_values(p.m)
        return result
    end
    result["point"] = Dict{String,Any}(k=>number.(v) for (k,v) in p.point)
    result["point"]["cost"] = number(p.cost)
    result["lambda"], result["mu"] = -dual.(p.href), abs.(dual.(p.gref))
    result["primal_error"] = max(maximum(abs.(number.(p.h)); init=0.0), maximum(number.(p.g); init=0.0))
    result["success"] = result["primal_error"] <= request["tolerance"]
    return result
end

function final_opf(request)
    data = PowerModels.parse_file(request["casefile"])
    net = request["network"]
    by_bus = Dict(bus=>i for (i,bus) in enumerate(net["load_bus"]))
    for (_, load) in data["load"]
        i = by_bus[load["load_bus"]]
        load["pd"], load["qd"] = request["pd"][i], request["qd"][i]
    end
    result = PowerModels.solve_opf(data, ACPPowerModel, optimizer(request["seconds"]))
    record = Dict{String,Any}("success"=>result["termination_status"] in (MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED),
        "status"=>string(result["termination_status"]), "solve_seconds"=>result["solve_time"])
    if !record["success"]
        return record
    end
    s = result["solution"]
    point = Dict{String,Any}("pd"=>request["pd"], "qd"=>request["qd"], "cost"=>result["objective"])
    for (key, table, field, n) in (("theta", "bus", "va", net["n"]), ("vm", "bus", "vm", net["n"]),
                                  ("pg", "gen", "pg", net["g"]), ("qg", "gen", "qg", net["g"]))
        point[key] = [s[table][string(i)][field] for i in 1:n]
    end
    record["point"] = point
    return record
end
