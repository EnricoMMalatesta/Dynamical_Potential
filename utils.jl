import GSL: sf_log_erfc
using SpecialFunctions
using LinearAlgebra

using ExtractMacro
import ForwardDiff
import ForwardDiff: Dual, value, partials
using Zygote
using QuadGK
using DelimitedFiles
using Parameters

using LinearAlgebra
import Base.abs, Base.max
#using CUDA
using Serialization: serialize, deserialize
using IterTools
using DelimitedFiles
using ThreadsX
using BasicInterpolators#: LinearInterpolator, BilinearInterpolator, BicubicInterpolator#, BicubicSplineInterpolator (this does not exists!)

using DataFrames, CSV
using Memoize
using Strided
using DataFrames

abstract type AbstractParams end

############### ORDER PARAMS  ################
mutable struct OrderParams <: AbstractParams
	𝓒::Matrix{Float64}
    𝓒ᵈᵗ::Matrix{Float64}
    μ::Vector{Float64}
    μh::Vector{Float64}
    μf::Float64
    δμf::Float64
    ε::Float64

    OrderParams(𝓒::Matrix{Float64}, 𝓒ᵈᵗ::Matrix{Float64}, μ::Vector{Float64}, μh::Vector{Float64}, μf::Float64, δμf::Float64, ε::Float64) = new(𝓒, 𝓒ᵈᵗ, μ, μh, μf, δμf, ε)

    """ Zero Initialization"""
    function OrderParams(n::Int64)
        nint = n - 1
        𝓒   = zeros(2(2n+1), 2(2n+1))
        𝓒ᵈᵗ = zeros(2(2n+1), 2(2n+1))
        μ   = zeros(2nint+1)
        μh  = zeros(2nint+1)
        μf = 0.
        δμf = 0.
        ε = 0.
        new(𝓒, 𝓒ᵈᵗ, μ, μh, μf, δμf, ε) 
    end

    """ Equilibrium Initialization"""
    function OrderParams(n::Int64, τf::Float64, β::Float64, p::Int64)
		@assert τf > 0

        τ = τf/2
        # tf = 2τ
        Δt = τ / n

        Δ = 10;
        N = Δ * (2n+1)
        ΔT = Δt / Δ

        Ceq = zeros(2N)
        μeq = 1 + 0.5 * β^2 * ∂f(1.0, p)

        Ceq[1] = 1.
        Ceq[2] = Ceq[1] - ΔT
        den = 1 + ΔT * (1 + 0.5 * β^2 * ∂f(Ceq[1], p))

        for i=3:2N
            ∂Ceq = Ceq[2:i-1] - Ceq[1:i-2]
            Ceq[i] = (Ceq[i-1] - 0.5 * β^2 * ΔT * ( ∂f.(reverse(Ceq[2:i-1]), p) ⋅ ∂Ceq - ∂f(Ceq[1], p) * Ceq[i-1])) / den
        end

        ∂Ceq = (Ceq[2:end] - Ceq[1:end-1]) / ΔT
        ∂²Ceq = (∂Ceq[3:end-1] - ∂Ceq[2:end-2]) / ΔT

        C_eq = UpperTriangular(vcat([circshift(Ceq[1:Δ:N], i-1)' for i=1:2n+1]...))
        C_eq += C_eq' - I

        R_eq = UpperTriangular(vcat([circshift(- 0.5 * ∂Ceq[1:Δ:N], i-1)' for i=1:2n+1]...))
        R_eq += R_eq' + 0.5 * ∂Ceq[1] * I

        X_eq = UpperTriangular(vcat([circshift(0.25 * ∂²Ceq[1:Δ:N], i-1)' for i=1:2n+1]...))
        X_eq += X_eq' - 0.5 * (∂²Ceq[1] - 0.5 * μeq + 1/Δt) * I

        𝓒 = [C_eq R_eq; copy(R_eq) X_eq]

        temp = Ceq[1:Δ:end]
        C_eqdt = [temp[i+j-1] for i=1:2n+1, j=1:2n+1]
        temp = ∂Ceq[1:Δ:end]
        R_eqdt = [-0.5 * temp[i+j-1] for i=1:2n+1, j=1:2n+1]
        temp = ∂²Ceq[1:Δ:end]
        X_eqdt = [0.25 * temp[i+j-1] for i=1:2n+1, j=1:2n+1]

        𝓒ᵈᵗ = [C_eqdt R_eqdt; copy(R_eqdt) X_eqdt]

        nint = n-1
        μ = μeq * ones(2nint+1)
        μh = zeros(2nint+1)
        μf = μeq
        δμf = 0.
        ε = 0.
        
		new(𝓒, 𝓒ᵈᵗ, μ, μh, μf, δμf, ε)
	end

end

# Arithmetic operations on the struct OrderParams
function abs(x::OrderParams)
    return OrderParams(abs.(x.𝓒), abs.(x.𝓒ᵈᵗ), abs.(x.μ), abs.(x.μh), abs(x.μf), abs(x.δμf), abs(x.ε))
end

function max(x::OrderParams)
    return max(x.𝓒..., x.𝓒ᵈᵗ..., x.μ..., x.μh..., x.μf, x.δμf, x.ε)
end

function Base.:+(x::OrderParams, y::OrderParams)
    return OrderParams(x.𝓒+y.𝓒, x.𝓒ᵈᵗ+y.𝓒ᵈᵗ, x.μ+y.μ, x.μh+y.μh, x.μf+y.μf, x.δμf+y.δμf, x.ε+y.ε)
end

function Base.:-(x::OrderParams, y::OrderParams)
    return OrderParams(x.𝓒-y.𝓒, x.𝓒ᵈᵗ-y.𝓒ᵈᵗ, x.μ-y.μ, x.μh-y.μh, x.μf-y.μf, x.δμf-y.δμf, x.ε-y.ε)
end

function Base.:-(x::OrderParams)
    return OrderParams(-x.𝓒, -x.𝓒ᵈᵗ, -x.μ, -x.μh, -x.μf, -x.δμf, -x.ε)
end
function Base.:*(a::T, x::OrderParams) where {T<:Number}
    return OrderParams(a * x.𝓒, a * x.𝓒ᵈᵗ, a * x.μ, a * x.μh, a * x.μf, a * x.δμf, a * x.ε)
end
function Base.:*(x::OrderParams, a::T) where {T<:Number}
    return a * x
end
function Base.:/(x::OrderParams, a::T) where {T<:Number}
    return OrderParams(x.𝓒 / a, x.𝓒ᵈᵗ / a, x.μ / a, x.μh / a, x.μf / a, x.δμf / a, x.ε / a)
end


############### EXTERNAL PARAMS  ################
@with_kw mutable struct ExtParams <: AbstractParams
    p::Int64    = 3                # p-spin
    β::Float64  = 1.695            # inverse of temperature of the starting (equilibrium) configuration
    β2::Float64 = 1.695            # inverse of temperature of the final configuration
    τf::Float64 = 10.              # max time of integration
    n::Int64    = 100              # number of grid points
    Δt::Float64 = τf / (2 * n)     # integration step
    S::Float64  = 0.65             # overlap with starting configuration
    
    ExtParams(p::Int64, β::Float64, β2::Float64, τf::Float64, n::Int64, Δt::Float64, S::Float64) = new(p, β, β2, τf, n, Δt, S)

    ExtParams(v::Vector{Any}) = new(v[1], v[2], v[3], v[4], v[5], v[6], v[7])
end

@with_kw mutable struct Params <: AbstractParams
    nKrylov::Int64    = 100      # dimension of the Krylov base
    ϵKrylov::Float64  = 0.       # stop criterium for Krylov  
    ϵ::Float64        = 1e-4     # stop criterium
    ψ::Float64        = 0.       # damping
    maxiters::Int64   = 1000     # maximum number of iterations
    dlossy::Float64   = 3.       # controls the number of diagonals that we maintain: [0, dlossy]
    lossyscale::Int64 = 2        # amount of compression
    verb::Int64       = 2        # verbose
end

# THERMODYNAMIC FUNCTIONS, TODO
# mutable struct ThermFunc <: AbstractParams
#     φFP::Float64        # Franz-Parisi Potential
#     ε::Float64          # ε: derivative of the FP potential
#     e::Vector{Float64}  # energy
# end


f(x, p) = x^p          # p-spin f
∂f(x, p) = p * x^(p-1)
∂∂f(x, p) = p * (p-1) * x^(p-2)
∂∂∂f(x, p) = p * (p-1) * (p-2) * x^(p-3)
∂∂∂∂f(x, p) = p == 3 ? 0 : p * (p-1) * (p-2) * (p-3) * x^(p-4)


function plateau(β, p)
    ok, qEA = newton(q -> 0.5 * β^2 * ∂f(q, p)/q * (1-q) - 1, 0.5)
    if ok 
        return qEA
    else 
        return ok
    end
end

########## Useful function for extracting and inserting boundaries and bulk of matrices ##########
function extract_col(𝓒, n; col = 1)
    return vcat(𝓒[2:2n, col], 𝓒[2n+3:end-1, col])
end

function extract_cols(𝓒, n, args...)
    res = extract_col(𝓒, n; col = args[1])
    for i in args[2:end]
        res = hcat(res, extract_col(𝓒, n; col = i))
    end

    return res
end

@views function extract_bulk(𝓒, n)
    return [ 𝓒[2:2n, 2:2n] 𝓒[2:2n, 2n+3:end-1]; 𝓒[2n+3:end-1, 2:2n] 𝓒[2n+3:end-1, 2n+3:end-1] ] 
    # nint = n - 1
    # return reshape( CatView( ( extract_col(𝓒, n; col=j) for j in chain(2:2n, 2n+3:2(2n+1)-1) )...), 2(2nint+1), 2(2nint+1))
end


@views function insert_row!(eq, a, n, nint; row = 1)
    eq[row, 2:2n] .= a[1:2nint+1]
    eq[row, 2n+3:end-1] .= a[2nint+2:end]
end

@views function insert_boundaries!(eq, eqCR1i, eqCR1f, eqR2Xi, eqR2Xf, n, nint)

    insert_row!(eq, eqCR1i, n, nint; row = 1)
    eq[:,1] .= eq[1, :]

    insert_row!(eq, eqCR1f, n, nint; row = 2n+1)
    eq[:,2n+1] .= eq[2n+1, :]

    insert_row!(eq, eqR2Xi, n, nint; row = 2n+2)
    eq[:,2n+2] .= eq[2n+2, :]

    insert_row!(eq, eqR2Xf, n, nint; row = 2(2n+1))
    eq[:,2(2n+1)] .= eq[2(2n+1), :]

end

@views function insert_bound!(eq, eqbound, n, nint)
    insert_boundaries!(eq, eqbound[:,1], eqbound[:,2], eqbound[:,3], eqbound[:,4], n, nint)
end


@views function insert_bulk!(eq, eqint, n, nint)
    eq[2:2n, 2:2n] .= eqint[1:2nint+1, 1:2nint+1]                    #upper left block
    eq[2:2n, 2n+3:end-1] .= eqint[1:2nint+1, 2nint+2:end]            #upper right block
    eq[2n+3:end-1, 2:2n] .= eqint[2nint+2:end, 1:2nint+1]            #bottom left block
    eq[2n+3:end-1, 2n+3:end-1] .= eqint[2nint+2:end, 2nint+2:end]    #bottom right block
end


########## Resampling the solution ##########
function resample(𝓒::Matrix{Float64}, n, nn)
    x = collect(1:2n+1)
    𝓒i = BilinearInterpolator(x, x, 𝓒);
    𝓒n = [𝓒i(1 + ((2n+1)-1)/((2nn+1)-1) * i, 1 + ((2n+1)-1)/((2nn+1)-1) * j) for i=0:(2nn+1)-1, j=0:(2nn+1)-1]
    return 𝓒n
end

function resample(μ::Vector{Float64}, n, nn)
    x = collect(1:2n+1)
    μi = LinearInterpolator(x, μ);
    μn = [μi(1 + ((2n+1)-1)/((2nn+1)-1) * i) for i=0:(2nn+1)-1]
    return μn
end

function full_resample(op::OrderParams, ep::ExtParams, nn)
    @extract ep: n τf Δt β p S
    nint = n-1
    nnint = nn-1
    Δtn = τf / (2nn)

    C  = op.𝓒[1:2n+1, 1:2n+1]
    R̂2 = op.𝓒[1:2n+1, 2n+2:end]
    R̂1 = op.𝓒[2n+2:end, 1:2n+1]
    X̂  = op.𝓒[2n+2:end, 2n+2:end]

    Cn  = resample(C, n, nn)
    R̂2n = resample(R̂2, n, nn)
    R̂1n = resample(R̂1, n, nn)
    X̂n  = resample(X̂ + I / (2*Δt), n, nn) - I / (2*Δtn)

    Cdt  = op.𝓒ᵈᵗ[1:2n+1, 1:2n+1]
    R̂2dt = op.𝓒ᵈᵗ[1:2n+1, 2n+2:end]
    R̂1dt = op.𝓒ᵈᵗ[2n+2:end, 1:2n+1]
    X̂dt  = op.𝓒ᵈᵗ[2n+2:end, 2n+2:end]

    Cdtn  = resample(Cdt, n, nn)
    R̂2dtn = resample(R̂2dt, n, nn)
    R̂1dtn = resample(R̂1dt, n, nn)
    X̂dtn  = resample(X̂dt, n, nn)

    μn = resample(op.μ, nint, nnint)
    μhn = resample(op.μh, nint, nnint)

    for i=1:2nn+1
        Cn[i, i] = 1.
        R̂2n[i, i] = 0.5
        R̂1n[i, i] = 0.5
    end

    opn = OrderParams([Cn R̂2n; R̂1n X̂n], [Cdtn R̂2dtn; R̂1dtn X̂dtn], μn, μhn, op.μf, op.δμf, op.ε)
    
    opn.𝓒[2nn+2, 2nn+2] = 1/4. + β^2 * ∂f(1, p) / 8. - 1/(2Δtn)
    opn.𝓒ᵈᵗ[2nn+2, 2nn+2] = 1/4. + β^2 * ∂f(1, p) / 8.

    opn.𝓒[1, 2nn+1] = S
    opn.𝓒[2nn+1, 1] = S
    opn.𝓒ᵈᵗ[1, 2nn+1] = S
    opn.𝓒ᵈᵗ[2nn+1, 1] = S

    opn.𝓒ᵈᵗ[1, 1] = 1.0
    opn.𝓒ᵈᵗ[1, 2nn+2] = 0.5
    opn.𝓒ᵈᵗ[2nn+2, 1] = 0.5

    halfsym!(opn)

    return opn

end

########## Scalar Product of two Krylov Vectors ##########
function scalar_product(x::Vector{Array{Float64, N} where N}, y::Vector{Array{Float64, N} where N})
    sum(x .⋅ y)
end

function scalar_product(x, y)
    sum(x .⋅ y)
end

########## Alternative Scalar Product of two Krylov Vectors ##########
function prod_arr(x::Vector{Array{Float64, N} where N}, y::Vector{Array{Float64, N} where N}, n)
    scalar_product(x,y) - 3 * (x[1][1,:]' *  y[1][1,:] + x[1][2n+1,:]' *  y[1][2n+1,:] + x[1][2n+2,:]' *  y[1][2n+2,:] + x[1][end,:]' *  y[1][end,:])
end

########## Function that symmetrizes an order parameter ##########
function halfsym!(op::OrderParams)
    op.𝓒 .+= op.𝓒'
    op.𝓒 .*= 0.5
    op.𝓒ᵈᵗ .+= op.𝓒ᵈᵗ'
    op.𝓒ᵈᵗ .*= 0.5

    return op
end

########## Compression and inverse compression functions ##########
########## In the case of no compression, this only converts an OrderParams in a Vector of Arrays ##########
function compress(𝓒, n)
    Ccomp = hcat(LowerTriangular(𝓒[1:2n+1, 1:2n+1]), zeros(2n+1))
    Xcomp = hcat(zeros(2n+1), UpperTriangular(𝓒[2n+2:end, 2n+2:end]))
    R = 𝓒[1:2n+1, 2n+2:end]

    Ccomp + Xcomp, R
end

function scale_vector(v::Vector{Float64}, scale)
    n = length(v)
    
    x = collect(0:1.0/(n-1):1)
    vi = LinearInterpolator(x, v);
    
    # nn = round(scale * (n+1)) - 1       # rimedio pecione per avere (molte volte) il numero divisibile per lossyscale

    nn = n + 1
    while scale < 1 && nn % (1/scale) != 0
        nn += 1
    end
    
    nn = scale * nn - 1

    return [vi(i) for i=0:1.0/(nn-1):1]
end

function scale_vector_cubic(v::Vector{Float64}, scale)
    n = length(v)
    
    x = collect(0:1.0/(n-1):1)
    vi = CubicInterpolator(x, v);

    nn = round(scale * (n-1)) + 1

    return [vi(i) for i=0:1.0/(nn-1):1]
end


function scale_matrix(C::Matrix{Float64}, scale)
    nr, nc = size(C)

    xr = collect(0:1.0/(nr-1):1)
    xc = collect(0:1.0/(nc-1):1)

    Ci = BilinearInterpolator(xr, xc, C);

    nnr = round(scale * (nr-1)) + 1
    nnc = round(scale * (nc-1)) + 1

    return [Ci(i, j) for i=0:1.0/(nnr-1):1, j=0:1.0/(nnc-1):1]

end

function scale_matrix_cubic(C::Matrix{Float64}, scale)
    nr, nc = size(C)

    xr = collect(0:1.0/(nr-1):1)
    xc = collect(0:1.0/(nc-1):1)

    Ci = BicubicInterpolator(xr, xc, C);

    nnr = round(scale * (nr-1)) + 1
    nnc = round(scale * (nc-1)) + 1

    return [Ci(i, j) for i=0:1.0/(nnr-1):1, j=0:1.0/(nnc-1):1]

end



function lossy_comp(𝓒, ep::ExtParams, pars::Params)
    @extract ep: n τf
    @extract pars: dlossy lossyscale

    @assert n % (lossyscale/2) == 0
    
    C = 𝓒[1:2n+1, 1:2n+1]
    X = 𝓒[2n+2:end, 2n+2:end]
    R = 𝓒[1:2n+1, 2n+2:end]

    #2n+1
    bcon = floor(Int, dlossy * 2n / τf)      # effective value of n inside the new interval [0, dlossy]

    #n should be multiple of lossyscale/2
    Ccomp = scale_matrix(C, 1.0/lossyscale)
    Xcomp = scale_matrix(X, 1.0/lossyscale)
    Rcomp = scale_matrix(R, 1.0/lossyscale)

    CXcomp = tril(Ccomp, -1) + triu(Xcomp, 1)

    Crot = zeros(2n+1, 2n+1) 
    Xrot = zeros(2n+1, 2n+1) 
    Rrot  = zeros(2n+1, 2n+1)

    for i in 1:2n+1
        Crot[i, :] = circshift(C[i, :], -i-bcon)
        Xrot[i, :] = circshift(X[i, :], -i-bcon)
        Rrot[i, :] = circshift(R[i, :], -i-bcon)
    end

    diagsC = Crot[:, end-bcon:end]
    diagsX = Xrot[:, end-bcon:end]
    diagsR = Rrot[:, end-2bcon:end]
    

    return CXcomp, Rcomp, diagsC, diagsX, diagsR
end


# old compression for lossyscale >=2 not working well
function lossy_comp_old(CX, R, ep::ExtParams, pars::Params)
    @extract ep: n τf
    @extract pars: dlossy lossyscale

    #2n+1
    bcon = floor(Int, dlossy * 2n / τf)      # effective value of n inside the new interval [0, dlossy]

    # For R and lossyscale = 2, 2bcon represents the number of points we take around the diagonal (including the diagonal itself)
    # For CX and lossyscale = 2, 2bcon-1 represents the number of points we take around the diagonal (including the 2 diagonals themselves of C and X)

    num_row = 2n % lossyscale

    CXrot = zeros(2n+1-num_row, 2n+2)
    # CXrot = zeros(2n+1, 2n+2)
    Rrot  = zeros(2n+1-num_row, 2n+1)

    for i in 1:2n+1-num_row
        CXrot[i, :] = circshift(CX[i, :], -i-bcon)
        Rrot[i, :]  = circshift(R[i, :], -i-bcon)
    end

    tot_col_C = 2n + 2 - 2bcon + 1        # tot columns of CX you want to compress! 
    tot_col_R = 2n + 1 - 2bcon            # tot columns of R you want to compress!

    while (tot_col_C - 1) % lossyscale != 0 
        tot_col_C += 1
    end

    while (tot_col_R - 1) % lossyscale != 0 
        tot_col_R += 1
    end

    CXdiag = CXrot[:, tot_col_C+1:end]
    CXout  = CXrot[:, 1:tot_col_C]
    Rdiag  = Rrot[:, tot_col_R+1:end]
    Rout   = Rrot[:, 1:tot_col_R]

    CXcomp = scale_matrix(CXout, 1.0/lossyscale)
    Rcomp = scale_matrix(Rout, 1.0/lossyscale)

    return CXdiag, CXcomp, CX[:, 1:lossyscale], CX[:, end-lossyscale:end], CX[end-2num_row+1:end, :], # why the 2? I think is a typo
           Rdiag, Rcomp, R[:, 1:lossyscale-1], R[:, end-lossyscale+1:end], R[end-num_row+1:end, :]
end


function lossy_compμ(μ, ep::ExtParams, pars::Params)
    @extract ep: n τf
    @extract pars: dlossy lossyscale

    return scale_vector(μ, 1.0/lossyscale)
end   


function conv(op::OrderParams, ep::ExtParams, pars::Params; compression = :half)
    @extract ep: n

    if compression == :half
        CX, R = compress(op.𝓒, n)
        CXdt, Rdt = compress(op.𝓒ᵈᵗ, n)

        return [CX, R, CXdt, Rdt, op.μ, op.μh, [op.μf], [op.δμf], [op.ε]]
    elseif compression == :lossy
        return [lossy_comp(op.𝓒, ep, pars)..., lossy_comp(op.𝓒ᵈᵗ, ep, pars)...,
                lossy_compμ(op.μ, ep, pars), lossy_compμ(op.μh, ep, pars), 
                [op.μf], [op.δμf], [op.ε]]
    elseif compression == :lossyold
        CX, R = compress(op.𝓒, n)
        CXdt, Rdt = compress(op.𝓒ᵈᵗ, n)
                
        return [lossy_comp_old(CX, R, ep, pars)..., lossy_comp_old(CXdt, Rdt, ep, pars)..., 
                lossy_compμ(op.μ, ep, pars), lossy_compμ(op.μh, ep, pars), 
                [op.μf], [op.δμf], [op.ε]]
    else
        return [op.𝓒, op.𝓒ᵈᵗ, op.μ, op.μh, [op.μf], [op.δμf], [op.ε]]
    end
end


function decompress(CX, R, n)
    cc = CX[1:2n+1, 1:2n+1]
    C = LowerTriangular(cc) +  tril(cc, -1)' 

    cx = CX[1:2n+1, 2:2n+2];
    X = UpperTriangular(cx) +  triu(cx, 1)'

    return [C R; R' X]
end


function lossy_decomp(c, ep::ExtParams, pars::Params)
    @extract ep: n τf
    @extract pars: dlossy lossyscale

    bcon = floor(Int, dlossy * 2 * n / τf);

    CXcomp, Rcomp, diagsC, diagsX, diagsR = c[1], c[2], c[3], c[4], c[5]

    maskC  = hcat(zeros(2n+1, 2n+1-bcon-1), diagsC)
    maskX  = hcat(zeros(2n+1, 2n+1-bcon-1), diagsX)
    maskR  = hcat(zeros(2n+1, 2n+1-2bcon-1), diagsR)

    maskdiagC = zeros(2n+1, 2n+1)
    maskdiagX = zeros(2n+1, 2n+1)
    maskdiagR = zeros(2n+1, 2n+1)

    for i in 1:2n+1
        maskdiagC[i, :] = circshift(maskC[i, :], i+bcon)
        maskdiagX[i, :] = circshift(maskX[i, :], i+bcon)
        maskdiagR[i, :] = circshift(maskR[i, :], i+bcon)
    end

    C = scale_matrix(tril(CXcomp, -1) + tril(CXcomp, -1)', lossyscale)
    X = scale_matrix(triu(CXcomp, 1) + triu(CXcomp, 1)', lossyscale)
    R = scale_matrix(Rcomp, lossyscale)

    outmask = maskdiagR .== 0

    C = outmask .* C + (maskdiagC + maskdiagC' - Diagonal(maskdiagC))
    X = outmask .* X + (maskdiagX + maskdiagX' - Diagonal(maskdiagX))
    R = outmask .* R + maskdiagR
    
    return [C R; R' X]

end

function lossy_decomp_old(c, ep::ExtParams, pars::Params)
    @extract ep: n τf
    @extract pars: dlossy lossyscale

    bcon = floor(Int, dlossy * 2 * n / τf);

    num_row = 2n % lossyscale

    CXdiag = c[1]
    CXout = scale_matrix(c[2], lossyscale)

    Rdiag = c[6]
    Rout = scale_matrix(c[7], lossyscale)

    CXrot = [CXout CXdiag]
    Rrot = [Rout Rdiag]

    CX = zeros(2n+1, 2n+2)
    R = zeros(2n+1, 2n+1)

    for i in 1:2n+1 - num_row
        CX[i, :] = circshift(CXrot[i, :], i+bcon)
        R[i, :] = circshift(Rrot[i, :], i+bcon)
    end

    CX[:, 1:lossyscale] = c[3]
    CX[:, end-lossyscale:end] = c[4]
    CX[end-2num_row+1:end, :] = c[5]        #same two???

    R[:, 1:lossyscale-1] = c[8]
    R[:, end-lossyscale+1:end] = c[9]
    R[end-num_row+1:end, :] = c[10]

    return CX, R
end


function lossy_decompμ(c, lossyscale)
    return scale_vector(c, lossyscale)
end


function iconv(c::Vector{Array{Float64, N} where N}, ep::ExtParams, pars::Params; compression = :half)
    @extract ep: n 
    @extract pars: lossyscale

    if compression == :half
        𝓒 = decompress(c[1], c[2], n)
        𝓒ᵈᵗ = decompress(c[3], c[4], n)
        
        return OrderParams(𝓒, 𝓒ᵈᵗ, c[5], c[6], c[7][1], c[8][1], c[9][1])
    elseif compression == :lossy
        𝓒 = lossy_decomp(c[1:5], ep, pars)
        𝓒ᵈᵗ = lossy_decomp(c[6:10], ep, pars)

        μ = lossy_decompμ(c[11], lossyscale)
        μh = lossy_decompμ(c[12], lossyscale)

        return OrderParams(𝓒, 𝓒ᵈᵗ, μ, μh, c[13][1], c[14][1], c[15][1])
    elseif compression == :lossyold
        CX, R = lossy_decomp_old(c[1:10], ep, pars)
        CXdt, Rdt = lossy_decomp_old(c[11:20], ep, pars)

        μ = lossy_decompμ(c[21], lossyscale)
        μh = lossy_decompμ(c[22], lossyscale)

        𝓒 = decompress(CX, R, n)
        𝓒ᵈᵗ = decompress(CXdt, Rdt, n)

        # return OrderParams(𝓒, 𝓒ᵈᵗ, μ, μh, c[27][1], c[28][1], c[29][1])
        return OrderParams(𝓒, 𝓒ᵈᵗ, μ, μh, c[23][1], c[24][1], c[25][1])
    else
        return OrderParams(c[1], c[2], c[3], c[4], c[5][1], c[6][1], c[7][1])
    end
end


#################### NEWTON ####################
"""
    type NewtonMethod <: AbstractRootsMethod
        dx::Float64
        maxiters::Int
        verb::Int
        atol::Float64
    end
Type containg the parameters for Newton's root finding algorithm.
The default parameters are:
    NewtonMethod(dx=1e-7, maxiters=1000, verb=0, atol=1e-10)
"""
mutable struct NewtonMethod
    dx::Float64
    maxiters::Int
    verb::Int
    atol::Float64
end

mutable struct NewtonParameters
    δ::Float64
    ϵ::Float64
    verb::Int
    maxiters::Int
end

NewtonMethod(; dx=1e-7, maxiters=1000, verb=0, atol=1e-10) =
                                    NewtonMethod(dx, maxiters, verb, atol)

function ∇!(∂f::Matrix, f::Function, x0, δ, f0, x1)
    n = length(x0)
    copy!(x1, x0)
    for i = 1:n
        x1[i] += δ
        ∂f[:,i] = (f(x1) - f0) / δ
        x1[i] = x0[i]
    end
end

∇(f::Function, x0::Real, δ::Real, f0::Real) = (f(x0 + δ) - f0) / δ

"""
    newton(f, x₀, pars=NewtonMethod())
Apply Newton's method with parameters `pars` to find a zero of `f` starting from the point
`x₀`.
The derivative of `f` is computed by numerical discretization. Multivariate
functions are supported.
Returns a tuple `(ok, x, it, normf)`.
**Usage Example**
ok, x, it, normf = newton(x->exp(x)-x^4, 1.)
ok || normf < 1e-10 || warn("Newton Failed")
"""
#note that in 1.0 warnings are eliminated at all
function newton(f, x₀::Float64, m=NewtonMethod())
    η = 1.0
    ∂f = 0.0
    x = x₀
    x1 = 0.0

    f0 = f(x)
    @assert isa(f0, Real)
    normf0 = abs(f0)
    it = 0
    while normf0 ≥ m.atol
        #m.verb > 1 && println("normf0 = $normf0, maximum precision = $(m.atol)")
        it > m.maxiters && return (false, x, it, normf0)
        it += 1
        if m.verb > 1
            println("(𝔫) it=$it")
            println("(𝔫)   x=$x")
            println("(𝔫)   f(x)=$f0")
            println("(𝔫)   normf=$(abs(f0))")
            println("(𝔫)   η=$η")
        end
        δ = m.dx
        while true
            try
                ∂f = ∇(f, x, δ, f0)
                break
            catch err
                #warn("newton: catched error:")
                #Base.display_error(err, catch_backtrace())
                δ /= 2
                #warn("new δ = $δ")
            end
            if δ < 1e-20
                #normf0 ≥ m.atol && warn("newton:  δ=$δ")
                println("Problema di δ!!")
                return (false, x, it, normf0)
            end
        end
        Δx = -f0 / ∂f
        m.verb > 1 && println("(𝔫)  Δx=$Δx")
        while true
            x1 = x + Δx * η
            local new_f0, new_normf0
            try
                new_f0 = f(x1)
                new_normf0 = abs(new_f0)
            catch err
                #warn("newton: catched error:")
                #Base.display_error(err, catch_backtrace())
                new_normf0 = Inf
            end
            if new_normf0 < normf0
                η = min(1.0, η * 1.1)
                f0 = new_f0
                normf0 = new_normf0
                x = x1
                break
            end
            # η is lowered if f(x1) fails, or if new_normf0 ≥ normf0
            η /= 2
            #η problem arises when the derivatives for example is ≈ 0 and x1 is really different from x and the new_normf0 ≫ normf0
            η < 1e-20 && println("Problema di η!!")
            η < 1e-20 && return (false, x, it, normf0)
        end
    end
    return true, x, it, normf0
end

function newton(f::Function, x₀, pars::NewtonParameters)
    η = 1.0
    n = length(x₀)
    ∂f = Array(Float64, n, n)
    x = Float64[x₀[i] for i = 1:n]  #order parameters
    x1 = Array(Float64, n)

    f0 = f(x)                       #system of equation
    @assert length(f0) == n
    @assert isa(f0, Union(Real,Vector))
    normf0 = vecnorm(f0)
    it = 0
    while normf0 ≥ pars.ϵ
        it > pars.maxiters && return (false, x, it, normf0)
        it += 1
        if pars.verb > 1
            println("(𝔫) it=$it")
            println("(𝔫)   x=$x")
            println("(𝔫)   f0=$f0")
            println("(𝔫)   norm=$(vecnorm(f0))")
            println("(𝔫)   η=$η")
        end
        δ = pars.δ
        while true
            try
                ∇!(∂f, f, x, δ, f0, x1)
                break
            catch
                δ /= 2
            end
            if δ < 1e-15
                normf0 ≥ pars.ϵ && warn("newton:  δ=$δ")
                return (false, x, it, normf0)
            end
        end
        if isa(f0, Vector)
            Δx = -∂f \ f0
        else
            Δx = -f0 / ∂f[1,1]
        end
        pars.verb > 1 && println("(𝔫)  Δx=$Δx")
        while true
            for i = 1:n
                x1[i] = x[i] + Δx[i] * η
            end
            local new_f0, new_normf0
            try
                new_f0 = f(x1)
                new_normf0 = vecnorm(new_f0)
            catch
                new_normf0 = Inf
            end
            if new_normf0 < normf0
                η = min(1.0, η * 1.1)
                if isa(f0, Vector)
                    copy!(f0, new_f0)
                else
                    f0 = new_f0
                end
                normf0 = new_normf0
                copy!(x, x1)
                break
            end
            η /= 2
            η < 1e-15 && return (false, x, it, normf0)
        end
    end
    return true, x, it, normf0
end
