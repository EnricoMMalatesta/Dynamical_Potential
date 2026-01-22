module DP

include("utils.jl")
BLAS.set_num_threads(16)


@memoize function computeΛ(𝓒, p, β, n)
    C  = @view 𝓒[1:2n+1, 1:2n+1]
    R̂2 = @view 𝓒[1:2n+1, 2n+2:end]
    R̂1 = @view 𝓒[2n+2:end, 1:2n+1]
    X̂  = @view 𝓒[2n+2:end, 2n+2:end]

    CΛ  = @. ∂f(C, p)
    R̂1Λ = @. ∂∂f(C, p) * R̂1
    R̂2Λ = @. ∂∂f(C, p) * R̂2
    X̂Λ  = @. ∂∂f(C, p) * X̂ + ∂∂∂f(C, p) * R̂1 * R̂2

    Λ = - β^2 * [X̂Λ R̂1Λ; R̂2Λ CΛ] / 2

    return Λ
end


computeΛ(op::OrderParams, ep::ExtParams) = computeΛ(op.𝓒, ep.p, ep.β, ep.n)
computeΛdt(op::OrderParams, ep::ExtParams) = computeΛ(op.𝓒ᵈᵗ, ep.p, ep.β, ep.n)

function computeM(μ, μh, n, Δt)
    nin = n - 1
    dv = - 2 * ones(2nin+1)
    ev = ones(2nin)
    ∂2 = - SymTridiagonal(dv, ev) / (2Δt^2)

    μb  = Diagonal(μ)
    μhb = Diagonal(μh)

    return [∂2 + μhb  μb; μb  -2I]#-2Diagonal(ones(2nin+1))]
end

computeM(op::OrderParams, ep::ExtParams) = computeM(op.μ, op.μh, ep.n, ep.Δt)

function compute_prod(𝓒, 𝓒ᵈᵗ, n, p, β, β2, Δt)

    Λ = computeΛ(𝓒, p, β, n)
    Λᵈᵗ = computeΛ(𝓒ᵈᵗ, p, β, n)

    prod = Λ * 𝓒 - Λᵈᵗ * 𝓒ᵈᵗ
    proddt = Λ * 𝓒ᵈᵗ + Λᵈᵗ * 𝓒 - 2 * Λᵈᵗ * 𝓒ᵈᵗ

    # corrections of order Δt² as given by the trapezoidal rule (plugging back the delta terms)
    prod -= 0.5 * (Λ[:, end] * 𝓒[end, :]' - Λᵈᵗ[:, end] * 𝓒ᵈᵗ[end, :]')
    prod[:, end] += 0.5 * Λ[:, end] * ( - 1 / (2Δt) )
    prod -= 0.5 * (Λ[:, 2n+1] * 𝓒[2n+1, :]' - Λᵈᵗ[:, 2n+1] * 𝓒ᵈᵗ[2n+1, :]')
    prod[2n+1, :] += 0.5 * (- β^2 * ∂∂f(1., p) / 2.) * (- 1 / (2Δt)) * 𝓒[2n+1, :]

    proddt -= 0.5 * (Λᵈᵗ[:, end] * 𝓒[end, :]' + Λ[:, end] * 𝓒ᵈᵗ[end, :]' - 2 * Λᵈᵗ[:, end] * 𝓒ᵈᵗ[end, :]')
    proddt[:, end] += 0.5 * Λᵈᵗ[:, end] * ( - 1 / (2Δt) )
    proddt -= 0.5 * (Λᵈᵗ[:, 2n+1] * 𝓒[2n+1, :]' + Λ[:, 2n+1] * 𝓒ᵈᵗ[2n+1, :]' - 2 * Λᵈᵗ[:, 2n+1] * 𝓒ᵈᵗ[2n+1, :]')
    proddt[2n+1, :] += 0.5 * (- β^2 * ∂∂f(1., p) / 2.) * (- 1 / (2Δt)) * 𝓒ᵈᵗ[2n+1, :]

    prod *= Δt
    proddt *= Δt

    # terms due to the initial conditions
    δpm = Λ[:, 2n+1+1] * 𝓒[1,:]' + β2 / (2β) * (Λ[:, 2(2n+1)] * 𝓒[2n+1,:]' - Λᵈᵗ[:, 2(2n+1)] * 𝓒ᵈᵗ[2n+1,:]')
    δpmdt = Λ[:, 2n+1+1] * 𝓒[1,:]' + β2 / (2β) * (Λ[:, 2(2n+1)] * 𝓒ᵈᵗ[2n+1,:]' + Λᵈᵗ[:, 2(2n+1)] * 𝓒[2n+1,:]' - 2 * Λᵈᵗ[:, 2(2n+1)] * 𝓒ᵈᵗ[2n+1,:]')

    prod += δpm
    proddt += δpmdt

    return prod, proddt
end

compute_prod(op::OrderParams, ep::ExtParams) = compute_prod(op.𝓒, op.𝓒ᵈᵗ, ep.n, ep.p, ep.β, ep.β2, ep.Δt)

function compute_eqbound(𝓒, 𝓒ᵈᵗ, 𝓒bound, 𝓒ᵈᵗbound, pbound, pbounddt, iM, n, Δt)
    nint = n-1

    v𝓒bound = zeros(2(2nint+1), 4)
    v𝓒bound[1, 1] = - 𝓒[1,1] / (2Δt^2)
    v𝓒bound[2nint+1, 1] = - 𝓒[2n+1,1] / (2Δt^2)

    v𝓒bound[1, 2] = - 𝓒[1, 2n+1] / (2Δt^2)
    v𝓒bound[2nint+1, 2] = - 𝓒[2n+1,2n+1] / (2Δt^2)

    v𝓒bound[1, 3] = - 𝓒[1, 2n+2] / (2Δt^2)
    v𝓒bound[2nint+1, 3] = - 𝓒[2n+1, 2n+2] / (2Δt^2)

    v𝓒bound[1, 4] = - 𝓒[1, 2(2n+1)] / (2Δt^2)
    v𝓒bound[2nint+1, 4] = - 𝓒[2n+1, 2(2n+1)] / (2Δt^2)

    v𝓒bounddt = zeros(2(2nint+1), 2)
    v𝓒bounddt[1, 1] = - 𝓒ᵈᵗ[1, 2n+1] / (2Δt^2)
    v𝓒bounddt[2nint+1, 1] = - 𝓒ᵈᵗ[2n+1, 2n+1] / (2Δt^2)

    v𝓒bounddt[1, 2] = - 𝓒ᵈᵗ[1, 2(2n+1)] / (2Δt^2)
    v𝓒bounddt[2nint+1, 2] = - 𝓒ᵈᵗ[2n+1, 2(2n+1)] / (2Δt^2)

    eqbound = - 𝓒bound - iM * (v𝓒bound + pbound)
    eqbounddt = - 𝓒ᵈᵗbound - iM * (v𝓒bounddt + pbounddt)

    eqbounddt = hcat(eqbound[:, 1], eqbounddt[:,1], eqbound[:, 3], eqbounddt[:, 2])

    return eqbound, eqbounddt
end

function compute_eqint(𝓒int, 𝓒ᵈᵗint, 𝓒bound, 𝓒ᵈᵗbound, prodint, proddtint, iM, n, Δt)
    nint = n-1

    bterm = iM[1,:] * 𝓒bound[:, 1]' + iM[2nint+1,:] * 𝓒bound[:, 2]'
    bterm = bterm/(2Δt^2)

    btermdt = iM[1,:] * 𝓒bound[:, 1]' + iM[2nint+1,:] * 𝓒ᵈᵗbound[:, 1]'
    btermdt = btermdt/(2Δt^2)

    eqint = - 𝓒int + iM / Δt + bterm - iM * prodint
    eqintdt = - 𝓒ᵈᵗint + btermdt - iM * proddtint

    return eqint, eqintdt
end

function insert_corners!(equa, 𝓒, 𝓒ᵈᵗ, μf, δμf, ε, prod, proddt, n, Δt; fixε = false)
    #eq B6 -> C(τ, τ)
    equa.𝓒[2n+1, 2n+1] = -2𝓒[2(2n+1), 2n+1] + μf * 𝓒[2n+1, 2n+1] + prod[2(2n+1), 2n+1]
    #eq B8 -> R2(τ, τ)
    equa.𝓒[2n+1, 2(2n+1)] = - 1/Δt - 2𝓒[2(2n+1), 2(2n+1)] + μf * 𝓒[2n+1, 2(2n+1)] + prod[2(2n+1), 2(2n+1)]
    #eq B10 -> Cᵈᵗ(τ, τ)
    equa.𝓒ᵈᵗ[2n+1, 2n+1] = -2𝓒ᵈᵗ[2(2n+1), 2n+1] + μf * 𝓒ᵈᵗ[2n+1, 2n+1] + proddt[2(2n+1), 2n+1]
    #eq B12 -> R2ᵈᵗ(τ, τ)
    equa.𝓒ᵈᵗ[2n+1, 2(2n+1)] = - 2𝓒ᵈᵗ[2(2n+1), 2(2n+1)] + μf * 𝓒ᵈᵗ[2n+1, 2(2n+1)] + proddt[2(2n+1), 2(2n+1)]

    #eq B15-B18 -> R1(τ, τ)
    equa.𝓒[2(2n+1), 2n+1] = (1.5 * 𝓒[2n+1, 2n+1] - 2𝓒[2n, 2n+1] + 0.5 * 𝓒[2n-1, 2n+1]) / Δt + 2𝓒[2(2n+1), 2n+1] + δμf * 𝓒[2n+1, 2n+1] - 2ε * 𝓒[1, 2n+1] - 2.
    # equa.𝓒[2(2n+1), 2n+1] = (11/6 * 𝓒[2n+1, 2n+1] - 3𝓒[2n, 2n+1] + 1.5 * 𝓒[2n-1, 2n+1] - 1/3 * 𝓒[2n-2, 2n+1]) / Δt + 2𝓒[2(2n+1), 2n+1] + δμf * 𝓒[2n+1, 2n+1] - 2ε * 𝓒[1, 2n+1] - 2.
    #eq B17-B18 -> X(τ, τ)
    equa.𝓒[2(2n+1), 2(2n+1)] = (1.5 * 𝓒[2n+1, 2(2n+1)] - 2𝓒[2n, 2(2n+1)] + 0.5 * 𝓒[2n-1, 2(2n+1)]) / Δt + 2*(𝓒[2(2n+1), 2(2n+1)] + 1/(2Δt)) + δμf * 𝓒[2n+1, 2(2n+1)] - 2ε * 𝓒[1, 2(2n+1)] - μf
    #eq B19a -> R1ᵈᵗ(τ, τ)
    equa.𝓒ᵈᵗ[2(2n+1), 2n+1] = (1.5 * 𝓒ᵈᵗ[2n+1, 2n+1] - 2𝓒ᵈᵗ[2n, 2n+1] + 0.5 * 𝓒ᵈᵗ[2n-1, 2n+1]) / Δt + 2𝓒ᵈᵗ[2(2n+1), 2n+1] + δμf * 𝓒ᵈᵗ[2n+1, 2n+1] - 2ε * 𝓒ᵈᵗ[1, 2n+1]
    # equa.𝓒ᵈᵗ[2(2n+1), 2n+1] = (11/6 * 𝓒ᵈᵗ[2n+1, 2n+1] - 3𝓒ᵈᵗ[2n, 2n+1] + 1.5 * 𝓒ᵈᵗ[2n-1, 2n+1] - 1/3 * 𝓒ᵈᵗ[2n-2, 2n+1]) / Δt + 2𝓒ᵈᵗ[2(2n+1), 2n+1] + δμf * 𝓒ᵈᵗ[2n+1, 2n+1] - 2ε * 𝓒ᵈᵗ[1, 2n+1]
    #eq B19b -> Xᵈᵗ(τ, τ)
    equa.𝓒ᵈᵗ[2(2n+1), 2(2n+1)] = (1.5 * 𝓒ᵈᵗ[2n+1, 2(2n+1)] - 2𝓒ᵈᵗ[2n, 2(2n+1)] + 0.5 * 𝓒ᵈᵗ[2n-1, 2(2n+1)]) / Δt + 2𝓒ᵈᵗ[2(2n+1), 2(2n+1)] + δμf * 𝓒ᵈᵗ[2n+1, 2(2n+1)] - 2ε * 𝓒ᵈᵗ[1, 2(2n+1)]

    if fixε
        #eq B6 -> C(τ, -τ), C(-τ, τ), Cᵈᵗ(τ, -τ), Cᵈᵗ(-τ, τ)
        equa.𝓒[2n+1, 1] = -2𝓒[2(2n+1), 1] + μf * 𝓒[2n+1, 1] + prod[2(2n+1), 1]
        equa.ε = 0.
    else
        #eq B6 -> becomes now the equation for epsilon (even if this below depends implicitly on ε)
        equa.𝓒[2n+1, 1] = 0.
        equa.ε = -2𝓒[2(2n+1), 1] + μf * 𝓒[2n+1, 1] + prod[2(2n+1), 1]
    end

    equa.𝓒[1, 2n+1] = equa.𝓒[2n+1, 1]
    equa.𝓒ᵈᵗ[2n+1, 1] = equa.𝓒[2n+1, 1]
    equa.𝓒ᵈᵗ[1, 2n+1] = equa.𝓒[2n+1, 1]

    #eq B8 -> R2(-τ, τ), R1(τ, -τ), R2ᵈᵗ(-τ, τ), R1ᵈᵗ(τ, -τ)
    equa.𝓒[1, 2(2n+1)] = - 2𝓒[2(2n+1), 2n+2] + μf * 𝓒[2n+1, 2n+2] + prod[2(2n+1), 2n+2]
    equa.𝓒[2(2n+1), 1] = equa.𝓒[1, 2(2n+1)]
    equa.𝓒ᵈᵗ[1, 2(2n+1)] = equa.𝓒[1, 2(2n+1)]
    equa.𝓒ᵈᵗ[2(2n+1), 1] = equa.𝓒[1, 2(2n+1)]

    #eq B15 -> R2(τ, -τ), R1(-τ, τ), R2ᵈᵗ(τ, -τ), R2ᵈᵗ(-τ, τ)
    equa.𝓒[2n+1, 2n+2] = (1.5 * 𝓒[2n+1, 1] - 2𝓒[2n, 1] + 0.5 * 𝓒[2n-1, 1]) / Δt + 2𝓒[2(2n+1), 1] + δμf * 𝓒[2n+1, 1] - 2ε * 𝓒[1, 1]
    equa.𝓒[2n+2, 2n+1] = equa.𝓒[2n+1, 2n+2]
    equa.𝓒ᵈᵗ[2n+1, 2n+2] = equa.𝓒[2n+1, 2n+2]
    equa.𝓒ᵈᵗ[2n+2, 2n+1] = equa.𝓒[2n+1, 2n+2]

    #eq B17 -> X(τ, -τ), X(-τ, τ), Xᵈᵗ(τ, -τ), Xᵈᵗ(-τ, τ)
    equa.𝓒[2(2n+1), 2n+2] = (1.5 * 𝓒[2n+1, 2n+2] - 2𝓒[2n, 2n+2] + 0.5 * 𝓒[2n-1, 2n+2]) / Δt + 2*𝓒[2(2n+1), 2n+2] + δμf * 𝓒[2n+1, 2n+2] - 2ε * 𝓒[1, 2n+2]
    equa.𝓒[2n+2, 2(2n+1)] = equa.𝓒[2(2n+1), 2n+2]
    equa.𝓒ᵈᵗ[2(2n+1), 2n+2] = equa.𝓒[2(2n+1), 2n+2]
    equa.𝓒ᵈᵗ[2n+2, 2(2n+1)] = equa.𝓒[2(2n+1), 2n+2]

end

"""

    Computes the saddle point equations given the dynamical correlation matrices. 
    
"""
function computeEquations(𝓒, 𝓒ᵈᵗ, μ, μh, μf, δμf, ε, n, p, β, β2, Δt; fixε = false)

    𝓒[2n+2, 2n+2] = 1/4. + β^2 * ∂f(1, p) / 8. - 1/(2Δt)
    𝓒ᵈᵗ[2n+2, 2n+2] = 1/4. + β^2 * ∂f(1, p) / 8.

    prod, proddt = compute_prod(𝓒, 𝓒ᵈᵗ, n, p, β, β2, Δt)

    # we extract the bulk of prod and proddt
    prodint = extract_bulk(prod, n)
    proddtint = extract_bulk(proddt, n)

    # we extract the boundaries of prod, for proddt we only need to extract the final conditions
    pbound = extract_cols(prod, n, 1, 2n+1, 2n+2, 2(2n+1))
    pbounddt = extract_cols(proddt, n, 2n+1, 2(2n+1))

    # we extract the bulk of 𝓒 and 𝓒ᵈᵗ
    𝓒int = extract_bulk(𝓒, n)
    𝓒ᵈᵗint = extract_bulk(𝓒ᵈᵗ, n)

    # we extract the boundaries of 𝓒, for 𝓒ᵈᵗ we only need the final condition
    𝓒bound = extract_cols(𝓒, n, 1, 2n+1, 2n+2, 2(2n+1))
    𝓒ᵈᵗbound = extract_cols(𝓒ᵈᵗ, n, 2n+1, 2(2n+1))

    # computation of the inverse of the M operator (only in the bulk!)
    M = computeM(μ, μh, n, Δt)
    iM = inv(Matrix(M))

    nint = n-1

    # computing the boundary and internal part of the equations
    eqbound, eqbounddt = compute_eqbound(𝓒, 𝓒ᵈᵗ, 𝓒bound, 𝓒ᵈᵗbound, pbound, pbounddt, iM, n, Δt)
    eqint, eqintdt = compute_eqint(𝓒int, 𝓒ᵈᵗint, 𝓒bound, 𝓒ᵈᵗbound, prodint, proddtint, iM, n, Δt)

    ##### CONSTRUCTING THE MATRIX OF EQUATIONS #####
    equa = OrderParams(n)

    insert_bound!(equa.𝓒, eqbound, n, nint)
    insert_bound!(equa.𝓒ᵈᵗ, eqbounddt, n, nint)

    insert_bulk!(equa.𝓒, eqint, n, nint)
    insert_bulk!(equa.𝓒ᵈᵗ, eqintdt, n, nint)

    insert_corners!(equa, 𝓒, 𝓒ᵈᵗ, μf, δμf, ε, prod, proddt, n, Δt; fixε = fixε)

    equa.μ  = 1 .- diag(𝓒int)[1:2nint+1]                            # C(t, t) == 1
    equa.μh = 1 .- diag(𝓒int, 2nint+1) .- diag(𝓒int, -(2nint+1))    # R1(t, t) + R2(t, t) == 1

    equa.μf = 1 - 𝓒[2n+1, 2n+1]                                     # C(τ, τ) == 1
    equa.δμf = 1 - 𝓒[2(2n+1), 2n+1] - 𝓒[2n+1, 2(2n+1)]              # R1(τ, τ) + R2(τ, τ) == 1

    return equa, prodint, proddtint, pbound, pbounddt, iM

end

computeEquations(op::OrderParams, ep::ExtParams; fixε = false) = computeEquations(op.𝓒, op.𝓒ᵈᵗ, op.μ, op.μh, op.μf, op.δμf, op.ε, ep.n, ep.p, ep.β, ep.β2, ep.Δt; fixε = fixε)


""" 
    
    Functor of the OrderParams struct, that computes the values attained by the dynamical equations.
    If 0 is returned the dynamical equations are exactly solved. 

"""
function (op::OrderParams)(ep::ExtParams; fixε = false)
    return computeEquations(op, ep; fixε = fixε)
end


function computeΔΛ(𝓒, Δ𝓒, p, β, n)
    C  = @view 𝓒[1:2n+1, 1:2n+1]
    R̂2 = @view 𝓒[1:2n+1, 2n+2:end]
    R̂1 = @view 𝓒[2n+2:end, 1:2n+1]
    X̂  = @view 𝓒[2n+2:end, 2n+2:end]

    ΔC  = @view Δ𝓒[1:2n+1, 1:2n+1]
    ΔR̂2 = @view Δ𝓒[1:2n+1, 2n+2:end]
    ΔR̂1 = @view Δ𝓒[2n+2:end, 1:2n+1]
    ΔX̂  = @view Δ𝓒[2n+2:end, 2n+2:end]

    ΔCΛ  = @. - 0.5 * β^2 * (∂∂f(C, p) * ΔC)
    ΔR̂1Λ = @. - 0.5 * β^2 * (∂∂∂f(C, p) * ΔC * R̂1 + ∂∂f(C, p) * ΔR̂1)
    ΔR̂2Λ = @. - 0.5 * β^2 * (∂∂∂f(C, p) * ΔC * R̂2 + ∂∂f(C, p) * ΔR̂2)
    ΔX̂Λ  = @. - 0.5 * β^2 * (∂∂∂f(C, p) * ΔC * X̂ + ∂∂f(C, p) * ΔX̂ + ∂∂∂∂f(C, p) * ΔC * R̂1 * R̂2 + ∂∂∂f(C, p) * ΔR̂1 * R̂2 + ∂∂∂f(C, p) * R̂1 * ΔR̂2)

    ΔΛ = [ΔX̂Λ ΔR̂1Λ; ΔR̂2Λ ΔCΛ]

    return ΔΛ
end

computeΔΛ(q::OrderParams, op::OrderParams, ep::ExtParams) = computeΔΛ(op.𝓒, q.𝓒, ep.p, ep.β, ep.n)
computeΔΛdt(q::OrderParams, op::OrderParams, ep::ExtParams) = computeΔΛ(op.𝓒ᵈᵗ, q.𝓒ᵈᵗ, ep.p, ep.β, ep.n)

function computeΔM(q::OrderParams, ep::ExtParams)
    @extract ep: n Δt

    nin = n - 1
    μb  = Diagonal(q.μ)
    μhb = Diagonal(q.μh)

    return [μhb  μb; μb  zeros(2nin+1, 2nin+1)]
end


out_prod!(α, u, v, A, n) = BLAS.gemm!('N', 'N', α, u[:,[2n+1, end]], v[[2n+1, end], :], 1.0, A)
out_prod!(α, u, v, A, i, j) = BLAS.gemm!('N', 'N', α, u[:,[i]], v[[j], :], 1.0, A)
out_prod!(α, u, v, A, i1, i2, j1, j2) = BLAS.gemm!('N', 'N', α, u[:,[i1, i2]], v[[j1, j2], :], 1.0, A)
out_prod!(α, u, v, A) = BLAS.gemm!('N', 'N', α, u, v, 1.0, A)


function compute_Δprod(Δ𝓒, Δ𝓒ᵈᵗ, 𝓒, 𝓒ᵈᵗ, n, p, β, β2, Δt)

    Λ = computeΛ(𝓒, p, β, n)
    Λᵈᵗ = computeΛ(𝓒ᵈᵗ, p, β, n)
    ΔΛ = computeΔΛ(𝓒, Δ𝓒, p, β, n)
    ΔΛᵈᵗ = computeΔΛ(𝓒ᵈᵗ, Δ𝓒ᵈᵗ, p, β, n)


    hprod = ΔΛᵈᵗ * 𝓒ᵈᵗ + Λᵈᵗ * Δ𝓒ᵈᵗ
    Δprod = ΔΛ * 𝓒 + Λ * Δ𝓒 - hprod
    Δproddt = ΔΛ * 𝓒ᵈᵗ + Λ * Δ𝓒ᵈᵗ + ΔΛᵈᵗ * 𝓒 + Λᵈᵗ * Δ𝓒 - 2*hprod

    # corrections of order Δt² as given by the trapezoidal rule (plugging back the delta terms)
    out_prod!(-0.5, ΔΛ[:,[2n+1, end]], 𝓒[[2n+1, end], :], Δprod)
    out_prod!(-0.5, Λ[:,[2n+1, end]], Δ𝓒[[2n+1, end], :], Δprod)
    out_prod!(0.5, ΔΛᵈᵗ[:,[2n+1, end]], 𝓒ᵈᵗ[[2n+1, end], :], Δprod)
    out_prod!(0.5, Λᵈᵗ[:,[2n+1, end]], Δ𝓒ᵈᵗ[[2n+1, end], :], Δprod)

    Δprod[:, end] .+= 0.5 * ΔΛ[:, end] * ( - 1 / (2Δt) )
    Δprod[2n+1, :] .+= 0.5 * (- β^2 * ∂∂f(1., p) / 2.) * (- 1 / (2Δt)) * Δ𝓒[2n+1, :]

    out_prod!(-0.5, ΔΛᵈᵗ[:,[2n+1, end]], 𝓒[[2n+1, end], :], Δproddt)
    out_prod!(-0.5, Λᵈᵗ[:,[2n+1, end]], Δ𝓒[[2n+1, end], :], Δproddt)
    out_prod!(-0.5, ΔΛ[:,[2n+1, end]], 𝓒ᵈᵗ[[2n+1, end], :], Δproddt)
    out_prod!(-0.5, Λ[:,[2n+1, end]], Δ𝓒ᵈᵗ[[2n+1, end], :], Δproddt)
    out_prod!(1.0, ΔΛᵈᵗ[:,[2n+1, end]], 𝓒ᵈᵗ[[2n+1, end], :], Δproddt)
    out_prod!(1.0, Λᵈᵗ[:,[2n+1, end]], Δ𝓒ᵈᵗ[[2n+1, end], :], Δproddt)

  
    Δproddt[:, end] .+= 0.5 * ΔΛᵈᵗ[:, end] * ( - 1 / (2Δt) )
    Δproddt[2n+1, :] .+= 0.5 * (- β^2 * ∂∂f(1., p) / 2.) * (- 1 / (2Δt)) * Δ𝓒ᵈᵗ[2n+1, :]

    Δprod *= Δt
    Δproddt *= Δt

    # terms due to the initial conditions
    δ = zeros(2(2n+1), 2(2n+1))
    out_prod!(1.0, ΔΛ[:, [2n+1+1]], 𝓒[[1], :], δ)
    out_prod!(1.0, Λ[:, [2n+1+1]], Δ𝓒[[1], :], δ)

    δ2 = zeros(2(2n+1), 2(2n+1))
    out_prod!(-β2 / (2β), ΔΛᵈᵗ[:, [2(2n+1)]], 𝓒ᵈᵗ[[2n+1], :], δ2)
    out_prod!(-β2 / (2β), Λᵈᵗ[:, [2(2n+1)]], Δ𝓒ᵈᵗ[[2n+1], :], δ2)

    Δprod .+= δ + δ2
    Δproddt .+= δ + 2*δ2

    out_prod!(β2 / (2β), ΔΛ[:, [2(2n+1)]], 𝓒[[2n+1], :], Δprod)
    out_prod!(β2 / (2β), Λ[:, [2(2n+1)]], Δ𝓒[[2n+1], :], Δprod)
    out_prod!(β2 / (2β), ΔΛ[:, [2(2n+1)]], 𝓒ᵈᵗ[[2n+1], :], Δproddt)
    out_prod!(β2 / (2β), Λ[:, [2(2n+1)]], Δ𝓒ᵈᵗ[[2n+1], :], Δproddt)
    out_prod!(β2 / (2β), ΔΛᵈᵗ[:, [2(2n+1)]], 𝓒[[2n+1], :], Δproddt)
    out_prod!(β2 / (2β), Λᵈᵗ[:, [2(2n+1)]], Δ𝓒[[2n+1], :], Δproddt)

    return Δprod, Δproddt
end

compute_Δprod(q::OrderParams, op::OrderParams, ep::ExtParams) = compute_Δprod(q.𝓒, q.𝓒ᵈᵗ, op.𝓒, op.𝓒ᵈᵗ, ep.n, ep.p, ep.β, ep.β2, ep.Δt)

function compute_Δeqbound(𝓒, Δ𝓒, 𝓒ᵈᵗ, Δ𝓒ᵈᵗ, Δ𝓒bound, Δ𝓒ᵈᵗbound, pbound, Δpbound, pbounddt, Δpbounddt, iM, ΔiM, n, Δt)
    nint = n-1

    v𝓒bound = zeros(2(2nint+1), 4)
    v𝓒bound[1, 1] = - 𝓒[1,1] / (2Δt^2)
    v𝓒bound[2nint+1, 1] = - 𝓒[2n+1,1] / (2Δt^2)

    v𝓒bound[1, 2] = - 𝓒[1, 2n+1] / (2Δt^2)
    v𝓒bound[2nint+1, 2] = - 𝓒[2n+1,2n+1] / (2Δt^2)

    v𝓒bound[1, 3] = - 𝓒[1, 2n+2] / (2Δt^2)
    v𝓒bound[2nint+1, 3] = - 𝓒[2n+1, 2n+2] / (2Δt^2)

    v𝓒bound[1, 4] = - 𝓒[1, 2(2n+1)] / (2Δt^2)
    v𝓒bound[2nint+1, 4] = - 𝓒[2n+1, 2(2n+1)] / (2Δt^2)

    v𝓒bounddt = zeros(2(2nint+1), 2)
    v𝓒bounddt[1, 1] = - 𝓒ᵈᵗ[1, 2n+1] / (2Δt^2)
    v𝓒bounddt[2nint+1, 1] = - 𝓒ᵈᵗ[2n+1, 2n+1] / (2Δt^2)

    v𝓒bounddt[1, 2] = - 𝓒ᵈᵗ[1, 2(2n+1)] / (2Δt^2)
    v𝓒bounddt[2nint+1, 2] = - 𝓒ᵈᵗ[2n+1, 2(2n+1)] / (2Δt^2)


    Δv𝓒bound = zeros(2(2nint+1), 4)
    Δv𝓒bound[1, 1] = - Δ𝓒[1,1] / (2Δt^2)
    Δv𝓒bound[2nint+1, 1] = - Δ𝓒[2n+1,1] / (2Δt^2)

    Δv𝓒bound[1, 2] = - Δ𝓒[1, 2n+1] / (2Δt^2)
    Δv𝓒bound[2nint+1, 2] = - Δ𝓒[2n+1,2n+1] / (2Δt^2)

    Δv𝓒bound[1, 3] = - Δ𝓒[1, 2n+2] / (2Δt^2)
    Δv𝓒bound[2nint+1, 3] = - Δ𝓒[2n+1, 2n+2] / (2Δt^2)

    Δv𝓒bound[1, 4] = - Δ𝓒[1, 2(2n+1)] / (2Δt^2)
    Δv𝓒bound[2nint+1, 4] = - Δ𝓒[2n+1, 2(2n+1)] / (2Δt^2)

    Δv𝓒bounddt = zeros(2(2nint+1), 2)
    Δv𝓒bounddt[1, 1] = - Δ𝓒ᵈᵗ[1, 2n+1] / (2Δt^2)
    Δv𝓒bounddt[2nint+1, 1] = - Δ𝓒ᵈᵗ[2n+1, 2n+1] / (2Δt^2)

    Δv𝓒bounddt[1, 2] = - Δ𝓒ᵈᵗ[1, 2(2n+1)] / (2Δt^2)
    Δv𝓒bounddt[2nint+1, 2] = - Δ𝓒ᵈᵗ[2n+1, 2(2n+1)] / (2Δt^2)

    Δeqbound = - Δ𝓒bound - ΔiM * (v𝓒bound + pbound) - iM * (Δv𝓒bound + Δpbound)
    Δeqbounddt = - Δ𝓒ᵈᵗbound - ΔiM * (v𝓒bounddt + pbounddt) - iM * (Δv𝓒bounddt + Δpbounddt)

    Δeqbounddt = hcat(Δeqbound[:, 1], Δeqbounddt[:,1], Δeqbound[:, 3], Δeqbounddt[:, 2])

    return Δeqbound, Δeqbounddt
end


function compute_Δeqint(Δ𝓒, Δ𝓒ᵈᵗ, Δ𝓒int, Δ𝓒ᵈᵗint, 𝓒bound, Δ𝓒bound, 𝓒ᵈᵗbound, Δ𝓒ᵈᵗbound, prodint, Δprodint, proddtint, Δproddtint, iM, ΔiM, n, Δt)
    nint = n-1

    Δeqint = zeros(2*(2nint+1), 2*(2nint+1))
    Δeqintdt = zeros(2*(2nint+1), 2*(2nint+1))

    out_prod!(1.0 / (2Δt^2), 𝓒bound[:, [1, 2]], ΔiM[[1, 2nint+1], :], Δeqint)
    out_prod!(1.0 / (2Δt^2), Δ𝓒bound[:, [1, 2]], iM[[1, 2nint+1], :], Δeqint)

    out_prod!(1.0 / (2Δt^2), hcat(𝓒bound[:, 1], 𝓒ᵈᵗbound[:, 1]), ΔiM[[1, 2nint+1],:], Δeqintdt)
    out_prod!(1.0 / (2Δt^2), hcat(Δ𝓒bound[:, 1], Δ𝓒ᵈᵗbound[:, 1]), iM[[1, 2nint+1],:], Δeqintdt)

    Δeqint .+= - Δ𝓒int + ΔiM / Δt - ΔiM * prodint - iM * Δprodint
    Δeqintdt .+= - Δ𝓒ᵈᵗint - ΔiM * proddtint - iM * Δproddtint

    return Δeqint, Δeqintdt
end


function insert_Δcorners!(Δequa, 𝓒, Δ𝓒, 𝓒ᵈᵗ, Δ𝓒ᵈᵗ, μf, Δμf, δμf, Δδμf, ε, Δε, Δprod, Δproddt, n, Δt; fixε = false)
    #eq B6 -> C(τ, τ)
    Δequa.𝓒[2n+1, 2n+1] = -2Δ𝓒[2(2n+1), 2n+1] + Δμf * 𝓒[2n+1, 2n+1] + μf * Δ𝓒[2n+1, 2n+1] + Δprod[2(2n+1), 2n+1]
    #eq B8 -> R2(τ, τ)
    Δequa.𝓒[2n+1, 2(2n+1)] =- 2Δ𝓒[2(2n+1), 2(2n+1)] + Δμf * 𝓒[2n+1, 2(2n+1)] + μf * Δ𝓒[2n+1, 2(2n+1)] + Δprod[2(2n+1), 2(2n+1)]
    #eq B10 -> Cᵈᵗ(τ, τ)
    Δequa.𝓒ᵈᵗ[2n+1, 2n+1] = -2Δ𝓒ᵈᵗ[2(2n+1), 2n+1] + Δμf * 𝓒ᵈᵗ[2n+1, 2n+1] + μf * Δ𝓒ᵈᵗ[2n+1, 2n+1] + Δproddt[2(2n+1), 2n+1]
    #eq B12 -> R2ᵈᵗ(τ, τ)
    Δequa.𝓒ᵈᵗ[2n+1, 2(2n+1)] = - 2Δ𝓒ᵈᵗ[2(2n+1), 2(2n+1)] + Δμf * 𝓒ᵈᵗ[2n+1, 2(2n+1)] + μf * Δ𝓒ᵈᵗ[2n+1, 2(2n+1)] + Δproddt[2(2n+1), 2(2n+1)]

    #eq B15-B18 -> R1(τ, τ)
    Δequa.𝓒[2(2n+1), 2n+1] = (1.5 * Δ𝓒[2n+1, 2n+1] - 2Δ𝓒[2n, 2n+1] + 0.5 * Δ𝓒[2n-1, 2n+1]) / Δt + 2Δ𝓒[2(2n+1), 2n+1] + Δδμf * 𝓒[2n+1, 2n+1] + δμf * Δ𝓒[2n+1, 2n+1] - 2ε * Δ𝓒[1, 2n+1]
    # Δequa.𝓒[2(2n+1), 2n+1] = (11/6 * Δ𝓒[2n+1, 2n+1] - 3Δ𝓒[2n, 2n+1] + 1.5 * Δ𝓒[2n-1, 2n+1] - 1/3 * Δ𝓒[2n-2, 2n+1]) / Δt + 2Δ𝓒[2(2n+1), 2n+1] + Δδμf * 𝓒[2n+1, 2n+1] + δμf * Δ𝓒[2n+1, 2n+1] - 2ε * Δ𝓒[1, 2n+1]
    #eq B17-B18 -> X(τ, τ)
    Δequa.𝓒[2(2n+1), 2(2n+1)] = (1.5 * Δ𝓒[2n+1, 2(2n+1)] - 2Δ𝓒[2n, 2(2n+1)] + 0.5 * Δ𝓒[2n-1, 2(2n+1)]) / Δt + 2*Δ𝓒[2(2n+1), 2(2n+1)] + Δδμf * 𝓒[2n+1, 2(2n+1)] + δμf * Δ𝓒[2n+1, 2(2n+1)] - 2ε * Δ𝓒[1, 2(2n+1)] - Δμf
    #eq B19a -> R1ᵈᵗ(τ, τ)
    Δequa.𝓒ᵈᵗ[2(2n+1), 2n+1] = (1.5 * Δ𝓒ᵈᵗ[2n+1, 2n+1] - 2Δ𝓒ᵈᵗ[2n, 2n+1] + 0.5 * Δ𝓒ᵈᵗ[2n-1, 2n+1]) / Δt + 2Δ𝓒ᵈᵗ[2(2n+1), 2n+1] + Δδμf * 𝓒ᵈᵗ[2n+1, 2n+1] + δμf * Δ𝓒ᵈᵗ[2n+1, 2n+1] - 2ε * Δ𝓒ᵈᵗ[1, 2n+1]
    # Δequa.𝓒ᵈᵗ[2(2n+1), 2n+1] = (11/6 * Δ𝓒ᵈᵗ[2n+1, 2n+1] - 3Δ𝓒ᵈᵗ[2n, 2n+1] + 1.5 * Δ𝓒ᵈᵗ[2n-1, 2n+1]- 1/3 * Δ𝓒ᵈᵗ[2n-2, 2n+1]) / Δt + 2Δ𝓒ᵈᵗ[2(2n+1), 2n+1] + Δδμf * 𝓒ᵈᵗ[2n+1, 2n+1] + δμf * Δ𝓒ᵈᵗ[2n+1, 2n+1] - 2ε * Δ𝓒ᵈᵗ[1, 2n+1]
    #eq B19b -> Xᵈᵗ(τ, τ)
    Δequa.𝓒ᵈᵗ[2(2n+1), 2(2n+1)] = (1.5 * Δ𝓒ᵈᵗ[2n+1, 2(2n+1)] - 2Δ𝓒ᵈᵗ[2n, 2(2n+1)] + 0.5 * Δ𝓒ᵈᵗ[2n-1, 2(2n+1)]) / Δt + 2Δ𝓒ᵈᵗ[2(2n+1), 2(2n+1)] + Δδμf * 𝓒ᵈᵗ[2n+1, 2(2n+1)] + δμf * Δ𝓒ᵈᵗ[2n+1, 2(2n+1)] - 2ε * Δ𝓒ᵈᵗ[1, 2(2n+1)]

    if fixε
        #eq B6 -> C(τ, -τ), C(-τ, τ)
        Δequa.𝓒[2n+1, 1] = -2Δ𝓒[2(2n+1), 1] + Δμf * 𝓒[2n+1, 1] + μf * Δ𝓒[2n+1, 1] + Δprod[2(2n+1), 1]
        Δequa.ε = 0.
    else
        Δequa.𝓒[2(2n+1), 2n+1] += - 2Δε * 𝓒[1, 2n+1]
        Δequa.𝓒[2(2n+1), 2(2n+1)] += - 2Δε * 𝓒[1, 2(2n+1)]
        Δequa.𝓒ᵈᵗ[2(2n+1), 2n+1] += - 2Δε * 𝓒ᵈᵗ[1, 2n+1]
        Δequa.𝓒ᵈᵗ[2(2n+1), 2(2n+1)] += - 2Δε * 𝓒ᵈᵗ[1, 2(2n+1)]
        #eq B6 -> becomes now the equation for epsilon (even if this below depends implicitly on ε)
        Δequa.𝓒[2n+1, 1] = 0.
        Δequa.ε = -2Δ𝓒[2(2n+1), 1] + Δμf * 𝓒[2n+1, 1] + μf * Δ𝓒[2n+1, 1] + Δprod[2(2n+1), 1]
    end

    Δequa.𝓒[1, 2n+1] = Δequa.𝓒[2n+1, 1]
    Δequa.𝓒ᵈᵗ[2n+1, 1] = Δequa.𝓒[2n+1, 1]
    Δequa.𝓒ᵈᵗ[1, 2n+1] = Δequa.𝓒[2n+1, 1]

    #eq B8 -> R2(-τ, τ), R1(τ, -τ), R2ᵈᵗ(-τ, τ), R1ᵈᵗ(τ, -τ)
    Δequa.𝓒[1, 2(2n+1)] = - 2Δ𝓒[2(2n+1), 2n+2] + Δμf * 𝓒[2n+1, 2n+2] + μf * Δ𝓒[2n+1, 2n+2] + Δprod[2(2n+1), 2n+2]
    Δequa.𝓒[2(2n+1), 1] = Δequa.𝓒[1, 2(2n+1)]
    Δequa.𝓒ᵈᵗ[1, 2(2n+1)] = Δequa.𝓒[1, 2(2n+1)]
    Δequa.𝓒ᵈᵗ[2(2n+1), 1] = Δequa.𝓒[1, 2(2n+1)]

    #eq B15 -> R2(τ, -τ), R1(-τ, τ), R2ᵈᵗ(τ, -τ), R2ᵈᵗ(-τ, τ)
    Δequa.𝓒[2n+1, 2n+2] = (1.5 * Δ𝓒[2n+1, 1] - 2Δ𝓒[2n, 1] + 0.5 * Δ𝓒[2n-1, 1]) / Δt + 2Δ𝓒[2(2n+1), 1] + Δδμf * 𝓒[2n+1, 1] + δμf * Δ𝓒[2n+1, 1] - 2ε * Δ𝓒[1, 1]
    Δequa.𝓒[2n+2, 2n+1] = Δequa.𝓒[2n+1, 2n+2]
    Δequa.𝓒ᵈᵗ[2n+1, 2n+2] = Δequa.𝓒[2n+1, 2n+2]
    Δequa.𝓒ᵈᵗ[2n+2, 2n+1] = Δequa.𝓒[2n+1, 2n+2]

    #eq B17 -> X(τ, -τ), X(-τ, τ), Xᵈᵗ(τ, -τ), Xᵈᵗ(-τ, τ)
    Δequa.𝓒[2(2n+1), 2n+2] = (1.5 * Δ𝓒[2n+1, 2n+2] - 2Δ𝓒[2n, 2n+2] + 0.5 * Δ𝓒[2n-1, 2n+2]) / Δt + 2*Δ𝓒[2(2n+1), 2n+2] + Δδμf * 𝓒[2n+1, 2n+2] + δμf * Δ𝓒[2n+1, 2n+2] - 2ε * Δ𝓒[1, 2n+2]
    Δequa.𝓒[2n+2, 2(2n+1)] = Δequa.𝓒[2(2n+1), 2n+2]
    Δequa.𝓒ᵈᵗ[2(2n+1), 2n+2] = Δequa.𝓒[2(2n+1), 2n+2]
    Δequa.𝓒ᵈᵗ[2n+2, 2(2n+1)] = Δequa.𝓒[2(2n+1), 2n+2]

    if !fixε
        Δequa.𝓒[2n+1, 2n+2] += - 2Δε * 𝓒[1, 1]
        Δequa.𝓒[2n+2, 2n+1] = Δequa.𝓒[2n+1, 2n+2]
        Δequa.𝓒ᵈᵗ[2n+1, 2n+2] = Δequa.𝓒[2n+1, 2n+2]
        Δequa.𝓒ᵈᵗ[2n+2, 2n+1] = Δequa.𝓒[2n+1, 2n+2]

        Δequa.𝓒[2(2n+1), 2n+2] += - 2Δε * 𝓒[1, 2n+2]
        Δequa.𝓒[2n+2, 2(2n+1)] = Δequa.𝓒[2(2n+1), 2n+2]
        Δequa.𝓒ᵈᵗ[2(2n+1), 2n+2] = Δequa.𝓒[2(2n+1), 2n+2]
        Δequa.𝓒ᵈᵗ[2n+2, 2(2n+1)] = Δequa.𝓒[2(2n+1), 2n+2]
    end

end

"""

    Computes the Jacobian of the equations times the current value attained by the equations.
    Basically it is the differential of the function `computeEquations``.

"""
function compute_jacobian(q::OrderParams, op::OrderParams, ep::ExtParams, prodint, proddtint, pbound, pbounddt, iM; fixε = false)
    @extract op: 𝓒 𝓒ᵈᵗ μ μh μf δμf ε
    @extract ep: n p β Δt

    Δprod, Δproddt = compute_Δprod(q, op, ep)

    # we extract the bulk of Δprod and Δproddt
    Δprodint = extract_bulk(Δprod, n)
    Δproddtint = extract_bulk(Δproddt, n)

    # we extract the boundaries of Δprod, and Δproddt for the final condition
    Δpbound = extract_cols(Δprod, n, 1, 2n+1, 2n+2, 2(2n+1))
    Δpbounddt = extract_cols(Δproddt, n, 2n+1, 2(2n+1))

    # we extract the bulk of Δ𝓒 and Δ𝓒ᵈᵗ
    Δ𝓒int = extract_bulk(q.𝓒, n)
    Δ𝓒ᵈᵗint = extract_bulk(q.𝓒ᵈᵗ, n)

    # we extract the boundaries of Δ𝓒, and Δ𝓒ᵈᵗ for the final condition
    𝓒bound = extract_cols(𝓒, n, 1, 2n+1, 2n+2, 2(2n+1))
    Δ𝓒bound = extract_cols(q.𝓒, n, 1, 2n+1, 2n+2, 2(2n+1))
    𝓒ᵈᵗbound = extract_cols(𝓒ᵈᵗ, n, 2n+1, 2(2n+1))
    Δ𝓒ᵈᵗbound = extract_cols(q.𝓒ᵈᵗ, n, 2n+1, 2(2n+1))

    # computation of the inverse of the ΔM operator (only in the bulk!)
    ΔM = computeΔM(q, ep)
    ΔiM = - iM * ΔM * iM

    # computing the boundary and internal part of the Jacobian
    Δeqbound, Δeqbounddt = compute_Δeqbound(𝓒, q.𝓒, 𝓒ᵈᵗ, q.𝓒ᵈᵗ, Δ𝓒bound, Δ𝓒ᵈᵗbound, pbound, Δpbound, pbounddt, Δpbounddt, iM, ΔiM, n, Δt)
    Δeqint, Δeqintdt = compute_Δeqint(q.𝓒, q.𝓒ᵈᵗ, Δ𝓒int, Δ𝓒ᵈᵗint, 𝓒bound, Δ𝓒bound, 𝓒ᵈᵗbound, Δ𝓒ᵈᵗbound, prodint, Δprodint, proddtint, Δproddtint, iM, ΔiM, n, Δt)

    ##### CONSTRUCTING THE JACOBIAN OF THE EQUATIONS #####
    nint = n-1
    Δequa = OrderParams(n)

    insert_bound!(Δequa.𝓒, Δeqbound, n, nint)
    insert_bound!(Δequa.𝓒ᵈᵗ, Δeqbounddt, n, nint)

    insert_bulk!(Δequa.𝓒, Δeqint, n, nint)
    insert_bulk!(Δequa.𝓒ᵈᵗ, Δeqintdt, n, nint)

    insert_Δcorners!(Δequa, 𝓒, q.𝓒, 𝓒ᵈᵗ, q.𝓒ᵈᵗ, μf, q.μf, δμf, q.δμf, ε, q.ε, Δprod, Δproddt, n, Δt; fixε = fixε)

    Δequa.μ  = - diag(Δ𝓒int)[1:2nint+1]                             # C(t, t) == 1
    Δequa.μh = - diag(Δ𝓒int, 2nint+1) .- diag(Δ𝓒int, -(2nint+1))    # R1(t, t) + R2(t, t) == 1

    Δequa.μf = - q.𝓒[2n+1, 2n+1]                                     # C(τf, τf) == 1
    Δequa.δμf = - q.𝓒[2(2n+1), 2n+1] - q.𝓒[2n+1, 2(2n+1)]            # R1(τf, τf) + R2(τf, τf) == 1

    return Δequa

end

"""
    Generalized minimal residual (GMRES) method routine

    GMRES is used to find an (approximate) solution of a large linear system of equations which in our case
    are of the form:
    
    J(op) * d(op) = - F(op)

    where `F(op)` is the vector of dynamical equations, `J(op)` its Jacobian and `d(op)` is the unknown 
    Newton step. The function returns the Newton's update d(op) = - J(op)⁻¹ * F(op). Note that 
    only Jacobian–vector products are evaluated. 

    The precision of the approximation depends on the size of the Krylov basis (stored in pars.nKrylov). 

    To reduce memory usage a compression of the dynamical matrices can be also used via the flag `compression`. 
    The compression parameters (`dlossy` and `lossyscale`) are stored in the struct pars. 

"""
function GMRES(op::OrderParams, ep::ExtParams, pars::Params; compression = :lossy, fixε = false)
    @extract ep: n
    @extract pars: nKrylov ϵKrylov verb lossyscale

    equa, prodint, proddtint, pbound, pbounddt, iM = op(ep; fixε = fixε)
    halfsym!(equa)

    q = sizehint!(Vector{Array{Float64, N} where N}[], nKrylov+1)
    h = zeros(nKrylov+1, nKrylov)

    push!(q, conv(-equa, ep, pars; compression = compression))
    h0 = √scalar_product(q[1], q[1])
    @. q[1] /= h0

    res0 = h0
    res = res0
    local y::Vector{Float64}
    local keff::Int64

    for k=2:nKrylov+1
        Δ = 0.0
        ok = true
        verb > 2 && k%10==0 && println("Krylov it=$k")

        push!(q, conv(halfsym!(compute_jacobian(iconv(q[k-1], ep, pars; compression = compression), op, ep, prodint, proddtint, pbound, pbounddt, iM; fixε = fixε)), ep, pars; compression = compression))

        # Arnoldi iteration
        @inbounds for j=1:k-1
            h[j, k-1] = scalar_product(q[j], q[k])
            for (qk, qj) in zip(q[k], q[j])
                @. qk -= h[j, k-1] * qj
            end
        end

        h[k, k-1] = √scalar_product(q[k], q[k])
        @. q[k] /= h[k, k-1]

        βe = zeros(k)
        βe[1] = h0
        H = h[1:k, 1:k-1]

        # least square minimization, y is a vector of dimension k-1
        y = H \ βe

        Δk = norm(H * y - βe)
        residue = max(abs.(H * y - βe)...)

        verb > 3 && k%10==0 && println(" Δk/k^0.5 =$(Δk/k^0.5), max = $(residue))\n")
        # GC.gc() # use to force julia to free memory
        keff = k
        ok &= residue <= ϵKrylov
        ok && break         # if ok==true, exit
    end

    qsol = y[1] * q[1]

    @inbounds for k in 2:keff-1
        for (qsolk, qk) in zip(qsol, q[k])
            @. qsolk += y[k] * qk
        end
    end

    q = nothing

    return iconv(qsol, ep, pars; compression = compression), h[1:keff-1, 1:keff-1]
end

"""
Biconjugate gradient stabilized method:

takes as an input a matrix A and a vector b and it solves the problem:

A x = b
"""
function BICGSTAB(A, b; ε = 1e-4, maxiters = 1000, verb = 0)

    @assert size(A)[1] == size(A)[2]
    @assert length(b) == size(A)[1]

    n = length(b)

    r0 = copy(b)
    p = copy(r0)
    rold = copy(r0)
    rnew = copy(r0)

    x = zeros(n)

    for k=0:maxiters-1
        Δ = 0.0
        ok = true
        verb > 2 && k%10==0 && println("BICGSTAB it=$k")

        @show r0

        Ap = A * p
        α = (rold ⋅ r0) / (Ap ⋅ r0)
        s = rold - α * Ap
        As = A * s
        ω = (As ⋅ s) / (As ⋅ As)
        @show α, ω
        x = x + α * p + ω * s

        rnew = s - ω * As

        @show rnew
        @show rold

        residue = √(rnew ⋅ rnew)

        verb > 3 #=&& k%10==0=# && println(" residue = $(residue)\n")

        ok &= residue <= ε
        ok && break         # if ok==true, exit

        β = α / ω * (rnew ⋅ r0) / (rold ⋅ r0)
        p = rnew + β * (p - ω * Ap)

        rold = rnew

    end

    return x
end


"""
Biconjugate gradient stabilized method adapted to the equations of the dynamics
"""
function BICGSTAB(op::OrderParams, ep::ExtParams, pars::Params; compression = :half, fixε = false, init = OrderParams(n))
    @extract ep: n
    @extract pars: nKrylov ϵKrylov verb lossyscale

    equa, prodint, proddtint, pbound, pbounddt, iM = op(ep; fixε = fixε)
    halfsym!(equa)

    A(x) = conv(halfsym!(compute_jacobian(iconv(x, ep, pars; compression = compression), op, ep, prodint, proddtint, pbound, pbounddt, iM; fixε = fixε)), ep, pars; compression = compression)


    dequa = conv(init, ep, pars; compression = compression)
    r0 = conv(-equa - iconv(A(dequa), ep, pars; compression = compression), ep, pars; compression = compression)

    p = r0
    rold = r0
    rnew = r0

    residue = √scalar_product(r0, r0)

    Δeq = max(abs(iconv(r0, ep, pars; compression = compression)))

    verb > 3 && println(" residue = $(residue)\n")
    verb > 3 && println(" Δeq = $(Δeq)\n")

    for k=0:nKrylov-1
        Δ = 0.0
        ok = true
        verb > 2 && k%10==0 && println("BICGSTAB it=$k")

        Ap = A(p)
        α = scalar_product(rold, r0) / scalar_product(Ap, r0)
        s = rold - α * Ap
        As = A(s)
        ω = scalar_product(As, s) / scalar_product(As, As)

        dequa = dequa + α * p + ω * s
        rnew = s - ω * As

        residue = √scalar_product(rnew, rnew)

        verb > 3 && #=k%10==0 &&=# println(" residue = $(residue)\n")

        ok &= residue <= ϵKrylov
        ok && break         # if ok==true, exit

        β = α / ω * scalar_product(rnew, r0) / scalar_product(rold, r0)
        p = rnew + β * (p - ω * Ap)

        rold = rnew

    end

    return iconv(dequa, ep, pars; compression = compression)
end

"""
Different version of the Biconjugate gradient stabilized method
"""
function BICGSTAB2(op::OrderParams, ep::ExtParams, pars::Params; compression = :half, fixε = false, init = OrderParams(ep.n))
    @extract ep: n
    @extract pars: nKrylov ϵKrylov verb lossyscale

    equa, prodint, proddtint, pbound, pbounddt, iM = op(ep; fixε = fixε)
    halfsym!(equa)

    A(x) = conv(halfsym!(compute_jacobian(iconv(x, ep, pars; compression = compression), op, ep, prodint, proddtint, pbound, pbounddt, iM; fixε = fixε)), ep, pars; compression = compression)


    # dequa = conv(OrderParams(n), ep, pars; compression = compression)
    dequa = conv(init, ep, pars; compression = compression)

    r0 = conv(-equa - iconv(A(dequa), ep, pars; compression = compression), ep, pars; compression = compression)
    r0equa = conv(-equa, ep, pars; compression = compression)

    @show max(abs(iconv(A(dequa), ep, pars; compression = compression)))

    ρold = 1.
    ρnew = 1.
    α = 1.
    ω = 1.

    ν = conv(OrderParams(n), ep, pars; compression = compression)
    p = copy(ν)

    r = copy(r0)


    residue = √scalar_product(r0, r0)
    Δeq = max(abs(iconv(r0, ep, pars; compression = compression)))
    verb > 3 && println(" residue = $(residue)\n")
    verb > 3 && println(" Δeq = $(Δeq)\n")

    for k=1:nKrylov
        Δ = 0.0
        ok = true
        verb > 2 && k%10==0 && println("BICGSTAB it=$k")

        ρnew = scalar_product(r0, r)
        β = ρnew / ρold * α / ω
        p = r + β * (p - ω * ν)
        ν = A(p)
        α = ρnew / scalar_product(r0, ν)

        h = dequa + α * p

        res1 = r0equa - A(h)
        residue1 = √scalar_product(res1, res1)
        verb > 3 && #=k%10==0 &&=# println(" residue1 = $(residue1)\n")

        ok &= residue1 <= ϵKrylov
        if ok
            dequa = h
            break
        end

        ok = true

        s = r - α * ν
        t = A(s)
        ω = scalar_product(t, s) / scalar_product(t, t)
        dequa = h + ω * s

        res2 = r0equa - A(dequa)
        residue2 = √scalar_product(res2, res2)
        verb > 3 && #=k%10==0 &&=# println(" residue2 = $(residue2)\n")
        ok &= residue2 <= ϵKrylov
        ok && break

        r = s - ω * t

        ρold = ρnew

    end

    return iconv(dequa, ep, pars; compression = compression)
end


###########################################################################
"""

    Function that for given external parameters `ep`` modifies the `OrderParams` struct `op` until the 
    dynamical equations are satisfied with a given degree of accuracy. 

"""
function converge!(op::OrderParams, ep::ExtParams, pars::Params; compression = :lossy, fixε = false)
    @extract pars: maxiters verb ϵ ψ

    Δ = Inf
    ok = false


    for it = 1:maxiters
        Δ = 0.0
        ok = true
        verb > 1 && println("it=$it")

        # before starting, we compress and decompress the initial condition.
        # this avoids having to deal with multiple solution to the problem
        # the solution that I reach does not depend anymore on the schedule that I use
        op = iconv(conv(op, ep, pars; compression = compression), ep, pars; compression = compression)

        # computation of the Newton's step update via Jacobian inversion
        global dequa, H = GMRES(op, ep, pars; compression = compression, fixε = fixε)
        # GC.gc()   # use to force julia to free memory

        # Newton's damped update
        op.𝓒 += (1 - ψ) * dequa.𝓒
        op.𝓒ᵈᵗ += (1 - ψ) * dequa.𝓒ᵈᵗ
        op.μ += (1 - ψ) * dequa.μ
        op.μh += (1 - ψ) * dequa.μh
        op.μf += (1 - ψ) * dequa.μf
        op.δμf += (1 - ψ) * dequa.δμf

        if !fixε
            op.ε += (1 - ψ) * dequa.ε
        end

        Δ = max(abs(dequa))
        equa = halfsym!(op(ep; fixε = fixε)[1])
        Δeq = max(abs(equa))

        verb > 1 && println(" Δ=$Δ")
        verb > 1 && println(" Δeq = $Δeq")

        ok &= Δ <= ϵ
        # ok &= Δeq <= ϵ
        ok && break         # if ok==true, exit
    end

    equa = halfsym!(op(ep; fixε = fixε)[1])

    return ok, Δ, op, equa, H
end


function converge(;
        p = 3, β = 0.5, β2 = 0.5, τf = 4., n = 100, S = 0.6,
        nKrylov = 50, ϵKrylov = 0.0, ϵ = 1e-4, ψ=0., maxiters = 1000, dlossy = 3.0, lossyscale = 2, verb = 2,
        kws...)

    τ = τf / 2
    Δt = τ / n
    nint = n - 1

    op = OrderParams(n, τf, β, p)
    ep = ExtParams(p, β, β2, τf, n, Δt, S)
    pars = Params(nKrylov, ϵKrylov, ϵ, ψ, maxiters, dlossy, lossyscale, verb)

    ok, Δ, op, equa, H = converge!(op, ep, pars; kws...)
    return ok, Δ, op, equa, H
end


"""
    Function that finds the solutions to the dynamical equations possibly for an iterator 
    over the external parameters:
        - max time of integration `τf`;
        - inverse temperature inverse of temperature of the starting (equilibrium) configuration `β`;
        - overlap with starting configuration `S`
        - number of grid points `n`.
"""
function span(;
        p = 3, β = 1.695, β2 = 1.695, τf = 10.0, n = 100, S = 0.65,
        nKrylov = 100, ϵKrylov = 0.0, ϵ = 1e-4, ψ=0., maxiters = 1000, dlossy = 3.0, lossyscale = 2, verb = 2,
        kws...)

    τ = first(τf) / 2
    Δt = τ / first(n)

    op = OrderParams(first(n), first(τf), first(β), p)
    ep = ExtParams(p, first(β), β2, first(τf), first(n), Δt, first(S))
    pars = Params(nKrylov, ϵKrylov, ϵ, ψ, maxiters, dlossy, lossyscale, verb)

    return span!(op, ep, pars; β=β, τf=τf, n=n, S=S, kws...)
end

function span!(op::OrderParams, ep::ExtParams, pars::Params; β=1.0, τf = 4., n=100, S = 0.6, ε = 0.,
                equalβ = true, compression = :lossy, save_file=true, fixε = false, dir = "./")

    if compression == :half
        pars.lossyscale = 0.
    end

    for β in β, τf in τf, n in n, S in S
        ep.β = β;
        if equalβ
            ep.β2 = ep.β
        end
        op = full_resample(op, ep, n)
        ep.τf = τf;
        ep.n = n;

        τ = ep.τf / 2
        ep.Δt = τ / ep.n;
        ep.S = S;
        if !fixε
            op.𝓒[1, 2n+1] = S
            op.𝓒[2n+1, 1] = S
            op.𝓒ᵈᵗ[1, 2n+1] = S
            op.𝓒ᵈᵗ[2n+1, 1] = S
        else
            op.ε = ε
        end

        println("# NEW ITER: p=$(ep.p)   β=$(ep.β)   β2=$(ep.β2)   τf=$(ep.τf)   n=$(ep.n)   S=$(ep.S)")

        @time ok, Δ, op, equa, H = converge!(op, ep, pars; compression = compression, fixε = fixε)        

        ok && save_file && open(dir * "FP_β$(ep.β)_n$(ep.n)_τf$(ep.τf)_S$(ep.S)_ls$(pars.lossyscale)_nK$(pars.nKrylov).txt", "w") do rf
            opcomp = conv(op, ep, pars; compression = :half)
            serialize(rf, (opcomp, ep, pars))
        end

    end

    return op, ep

end

"""
    Function that returns the starting equilibrium initialization.
"""
function equilibrium_init(;
    p = 3, β = 1.695, β2 = 1.695, τf = 5.0, n = 100, S = 0.65,
    nKrylov = 100, ϵKrylov = 0.0, ϵ = 1e-4, ψ=0., maxiters = 1000, dlossy = 3.0, lossyscale = 2, verb = 2,
    kws...)

    τ = first(τf) / 2
    Δt = τ / first(n)

    op = OrderParams(first(n), first(τf), first(β), p)
    ep = ExtParams(p, first(β), β2, first(τf), first(n), Δt, first(S))
    pars = Params(nKrylov, ϵKrylov, ϵ, ψ, maxiters, dlossy, lossyscale, verb)

    return op, ep, pars
end

"""
    span function from given initial conditions
"""
function span_from_ic(op::OrderParams, ep::ExtParams, pars::Params;
    p = ep.p, β = ep.β, β2 = ep.β2, τf = ep.τf, n = ep.n, Δt = ep.Δt, S = ep.S,
    nKrylov = pars.nKrylov, ϵKrylov = pars.ϵKrylov, ϵ = pars.ϵ, ψ = pars.ψ, maxiters = pars.maxiters,
    dlossy = pars.dlossy, lossyscale = pars.lossyscale, verb = pars.verb,
    kws...)

    # ep = ExtParams(p, first(β), β2, first(τf), first(n), Δt, first(S))
    pars = Params(nKrylov, ϵKrylov, ϵ, ψ, maxiters, dlossy, lossyscale, verb)

    return span!(op, ep, pars; β=β, τf=τf, n=n, S=S, kws...)
end


# UTILITY FUNCTIONS FOR SAVING/READ DATA
"""
Function that reads OrderParams and ExtParams from a serialized file
"""
function readparams(resfile)

    op, ep = open(resfile, "r") do io
        deserialize(io)
    end

    return op, ep
end

"""
Function that reads OrderParams, ExtParams, Params from a serialized file
"""
function readparams_all(resfile)

    opcomp, ep, pars = open(resfile, "r") do io
        deserialize(io)
    end

    op = iconv(opcomp, ep, pars; compression = :half)

    return op, ep, pars
end

"""
Function that reads OrderParams, ExtParams, Params from a uncompressed serialized file
"""
function readparams_all_uncomp(resfile)

    op, ep, pars = open(resfile, "r") do io
        deserialize(io)
    end

    return op, ep, pars
end

"""
Respan using an initial condition given from a file
"""
function respan(;
    p = 3, β = 1.695, β2 = 1.695, τf = 10.0, n = 100, S = 0.65:-0.05:0.0,
    nKrylov = 100, ϵKrylov = 0.0, ϵ = 1e-4, ψ=0., maxiters = 1000, dlossy = 3.0, lossyscale = 2, verb = 2,
    dir = "./",
    kws...)

    pars = Params(nKrylov, ϵKrylov, ϵ, ψ, maxiters, dlossy, lossyscale, verb)

    # for s in filter(contains("FP_β$(ep.β)_n$(ep.n)_τf$(ep.τf)_S$(ep.S)_ls$(pars.lossyscale)_nK$(pars.nKrylov).txt"), readdir(dir))

    for S in S, n in n
        op, ep = readparams(dir * "FP_β$(β)_n$(n)_τf$(τf)_S$(S)_ls$(lossyscale)_nK$(nKrylov).txt")

        span!(op, ep, pars; β=β, τf=τf, n=n, S=S, kws...)
    end

end


function output_dataframes(resfile)
    op, ep, pars = readparams_all(resfile)

    @extract ep: n
    df = DataFrame(S = Float64[], τf = Float64[], n = Int64[], t = Float64[], tp = Float64[], C = Float64[], R = Float64[], X = Float64[], Cdt = Float64[], Rdt = Float64[], Xdt = Float64[])
    dfμ = DataFrame(S = Float64[], τf = Float64[], n = Int64[], t = Float64[], μ = Float64[], μh = Float64[])

    t = 0:ep.Δt:ep.τf

    for i in 1:2n+1, j in 1:2n+1
        push!(df, (ep.S, ep.τf, ep.n, t[i], t[j], op.𝓒[i, j], op.𝓒[2n+1+i, j], op.𝓒[2n+1+i, 2n+1+j], op.𝓒ᵈᵗ[i, j], op.𝓒ᵈᵗ[2n+1+i, j], op.𝓒ᵈᵗ[2n+1+i, 2n+1+j]))
    end

    nin = n-1

    for i in 1:2nin+1
        push!(dfμ, (ep.S, ep.τf, ep.n, t[i+1], op.μ[i], op.μh[i]))
    end

    return df, dfμ
end

function add_data!(resfile, df, dfμ)
    op, ep, pars = readparams_all(resfile)

    @extract ep: n

    t = 0:ep.Δt:ep.τf

    for i in 1:2n+1, j in 1:2n+1
        push!(df, (ep.S, ep.τf, ep.n, t[i], t[j], op.𝓒[i, j], op.𝓒[2n+1+i, j], op.𝓒[2n+1+i, 2n+1+j], op.𝓒ᵈᵗ[i, j], op.𝓒ᵈᵗ[2n+1+i, j], op.𝓒ᵈᵗ[2n+1+i, 2n+1+j]))
    end

    nin = n-1

    for i in 1:2nin+1
        push!(dfμ, (ep.S, ep.τf, ep.n, t[i+1], op.μ[i], op.μh[i]))
    end

end



function readparams_csv(; Ctotfile = "Ctot.txt", Ctotdtfile = "Ctotdt.txt", μfile = "mu.txt", rest="rest.txt",
    μhfile = "muh.txt", epfile="ep.txt")

    𝓒 = Matrix{Float64}(CSV.read(Ctotfile, DataFrame, header=false))
    𝓒ᵈᵗ = Matrix{Float64}(CSV.read(Ctotdtfile, DataFrame, header=false))
    μ = Float64.(CSV.read(μfile, DataFrame, header=false)[:,1])
    μh = Float64.(CSV.read(μhfile, DataFrame, header=false)[:,1])
    r = Float64.(CSV.read(rest, DataFrame, header=false)[:,1])

    μf = r[1]
    δμf = r[2]
    ε = r[3]

    op = OrderParams(𝓒, 𝓒ᵈᵗ, μ, μh, μf, δμf, ε)
    ep = ExtParams(Vector{Any}(CSV.read(epfile, DataFrame, header=false)[:,1]))

    return op, ep
end



function print_data(; p = 3, β=1.695, n=100, τf=10.0, nKrylov = 800, S=0:0.05:0.65, lossyscale=2, dir = "./")

    results = []
    for p in S
        push!(results, readparams_all(dir * "FP_β$(β)_n$(n)_τf$(τf)_S$(p)_ls$(lossyscale)_nK$(nKrylov).txt"))
    end

    μeq = 1 + p * β^2 / 2
    nin = n - 1

    τfi = floor(Int64, τf)

    open("ep-$(β)-$(n)-$(τfi)", "w") do rf
        string = "{"
        for (i, p) in enumerate(S)
            string *= "{$p, $(results[i][1].ε)}, "
        end
        string = chop(string, tail=2)
        string *= "}"
        print(rf, string)
    end

    open("Cp-$(β)-$(n)-$(τfi)", "w") do rf
        string = "{"
        for (i, p) in enumerate(S)
            sμ = "{"

            for j in 1:2n+1
                sμ *= "{$(results[i][2].Δt * (j-1)), $(results[i][1].𝓒[1, j])}, "
            end

            sμ = chop(sμ, tail=2)
            sμ *= "}"
            string *= "{$p, " * sμ *"}, "

        end

        string = chop(string, tail=2)
        string *= "}"
        print(rf, string)
    end

    open("mup-$(β)-$(n)-$(τfi)", "w") do rf
        string = "{"
        for (i, p) in enumerate(S)
            sμ = "{"

            for j in 1:2nin+1
                sμ *= "{$(results[i][2].Δt * j), $(results[i][1].μ[j])}, "
            end

            sμ = chop(sμ, tail=2)
            sμ *= "}"
            string *= "{$p, " * sμ *"}, "

        end

        string = chop(string, tail=2)
        string *= "}"
        print(rf, string)
    end

    open("mutp-$(β)-$(n)-$(τfi)", "w") do rf
        string = "{"
        for (i, p) in enumerate(S)
            sμ = "{"

            for j in 1:2nin+1
                sμ *= "{$(results[i][2].Δt * j), $(results[i][1].μh[j])}, "
            end

            sμ = chop(sμ, tail=2)
            sμ *= "}"
            string *= "{$p, " * sμ *"}, "

        end

        string = chop(string, tail=2)
        string *= "}"
        print(rf, string)
    end

end


function Base.println(io::IO, v::Vector)

    for x in v
        print(io, x, " ")
    end

    println(io, )
end

function print_all(op::OrderParams, ep::ExtParams)
    @extract ep: n

    nint = n - 1

    for row = 1:2(2n+1)

        open("C_β$(ep.β)_n$(ep.n)_τf$(ep.τf)_S$(ep.S).txt", "a") do rf
            println(rf, op.𝓒[row, :])
        end

        open("Cdt_β$(ep.β)_n$(ep.n)_τf$(ep.τf)_S$(ep.S).txt", "a") do rf
            println(rf, op.𝓒ᵈᵗ[row, :])
        end

    end

    open("μ_β$(ep.β)_n$(ep.n)_τf$(ep.τf)_S$(ep.S).txt", "a") do rf
        println(rf, op.μ)
        println(rf, op.μh)
        println(rf, op.μf)
        println(rf, op.δμf)
        println(rf, op.ε)
    end

end

end ## module
