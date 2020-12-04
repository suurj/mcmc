using LinearAlgebra
using QuadGK
using SpecialFunctions
using PyPlot
using SparseArrays
using ProgressBars
using StatsBase
using AlphaStableDistributions
using AdvancedHMC
using Dates
using MAT
using Optim
using MCMCDiagnostics
using Dates

include("apu.jl")

function reweight_chain(samples,Npseudo,target,T)
    Ns = size(samples)[2]
    Ncompo = Int64(size(samples)[1]/Npseudo)
    R = zeros(Ncompo,Ns)

    iv = Vector(1:Npseudo)
    for i = 1:Ns
        w = zeros(Npseudo,)
        for j = 1:Npseudo
            w[j] = target(samples[(j-1)*Ncompo+1:j*Ncompo,i])*(1-1/T)
        end
        w .= exp.(w .- maximum(w))
        index = sample(iv,StatsBase.pweights(w))
        R[:,i] = samples[(index-1)*Ncompo+1:index*Ncompo,i]
    end

    return R
end


function atleast1d(x::Union{Real})
    p = zeros(1,) 
    p[1] = x
    return  p

end

function atleast1d(x::Vector{Float64})
    return  x

end

function leapfrog!(xnew,pnew,xinit,pinit,Mi,glpdf,epsilon,L)
    xnew .= xinit
    pnew .= pinit + epsilon/2*glpdf(xnew) 
    for i = 1:L
        xnew .= xnew + epsilon*Mi*pnew
        pnew .= pnew + epsilon*glpdf(xnew)
    end
    xnew .= xnew + epsilon*Mi*pnew
    pnew .= pnew + epsilon/2*glpdf(xnew)
    

    return nothing
end

function partialreplace(orig,replace,ix)
    a = copy(orig)
    a[ix] = replace
    return a
end

function  logpcauchy(W,arguments)
    Mat = arguments.Mat; 
    noisesigma = arguments.noisesigma
    scale = arguments.scale; y  = arguments.y; #F = arguments.F
    G = 1;

    r = Mat*(W);
    #r = F*U;

    logp = -0.5/noisesigma^2*sum((r-y).^2);
    logp = logp + sum(log.(scale./(pi*(scale^2.0.+W.^2))));

    return logp;

end

function logpcauchygradi(W,arguments;both=false)
    logp = 0.0
    Mat = arguments.Mat; 
    noisesigma = arguments.noisesigma
    scale = arguments.scale; y  = arguments.y; #F = arguments.F

    r = Mat*(W);
    #r = F*U;

    if(both)
        logp = -0.5/noisesigma^2*sum((r-y).^2);
        logp = logp + sum(log.(scale./(pi*(scale^2.0.+W.^2))));
    end

    Gr = Mat

    nt = length(y); N =size(W)[1];
    G = zeros(N,);
    G =  -1/noisesigma^2*Mat'*(r-y)
    # for i = 1:nt
    #     q =(-((r[i]-y[i])./noisesigma.^2))*Gr[i,:]';
    #     G = G + dropdims(q, dims = tuple(findall(size(q) .== 1)...))
    # end

   G = G  -2*W./(scale^2 .+ W.^2);
   return logp,G

end

function logpcauchyrep(w,args,cache)
    x=cache.noiseprop; 
    Mxprop = cache.Mxprop
    res = cache.residual
    noisevar = args.noisesigma^2
    scale = args.scale

    if(maximum(abs.(w)) > 100)
        return -Inf
    end

    y = args.y
    M = args.Mat
    x .= pi*(logsigmoid.(w)) .- pi/2
    logp = sum(w-2*log1pexp.(w))
    x .= scale*tan.(x)
    Mxprop = M*x
    res = Mxprop-y 
    logp = logp -0.5/noisevar*dotitself(res)

    return  logp
end


function logpcauchyrepgradi(w,args,cache;both=false)
    logp = 0.0
    #q = cache.c1; G=cache.c2; x=cache.c3; res = cache.c4
    x = cache.noiseprop
    res = cache.residual
    noisevar = args.noisesigma^2
    scale = args.scale
    M = args.Mat
    y = args.y
    x .= pi*(logsigmoid.(w)) .- pi/2
    x .= scale*tan.(x)
    G =  1 .- 2*logsigmoid.(w)
    q = @. (scale*pi*exp(w)*(tan((pi*(exp(w) - 1))/(2*(exp(w) + 1)))^2 + 1))/(exp(w) + 1)^2
    res .= y-M*x
    G .= G + diagm(q)/noisevar*M'*(res)
    if(both)
        logp = sum(w-2*log1pexp.(w))
        logp = logp -0.5/noisevar*dotitself(res)
    end
    return logp,G
 
end

function logpcauchyreppartial(ix,w,args,cache,current)
    Nix = length(ix)
    #w = SArray{Tuple{Nix,}}(worig[ix]) 
    res = cache.residual
    noisecurr = current.noisecurr
    Mxcurr = current.Mxcurr

    if(maximum(abs.(w)) > 100)
        return -Inf
    end

    Mxprop = cache.Mxprop
    noiseprop = cache.noiseprop

    currx = @view noisecurr[ix]
    noisevar = args.noisesigma^2
    scale = args.scale
    y = args.y
    M = @view args.Mat[:,ix]
    noiseprop[ix] .= pi*(logsigmoid.(w)) .- pi/2
    logp = sum(w-2*log1pexp.(w))

    noiseprop[ix] .= scale*tan.(noiseprop[ix])
    compx = noiseprop[ix]-currx
    Mxprop .= Mxcurr
    Mxprop .+= M*compx

    res .= Mxprop 
    res .-=  y
    logp = logp -0.5/noisevar*dotitself(res)

    return  logp
end

function logpcauchyreppartialgradi(ix,w,args,cache,current;both=false)
    logp = 0.0
    #q = cache.c1; G=cache.c2; x=cache.c3; 

    Mxprop = cache.Mxprop
    noiseprop = cache.noiseprop
    G = cache.gradiprop

    res = cache.residual
    noisevar = args.noisesigma^2
    scale = args.scale
    allcurrx = current.noisecurr
    Mxcurr = current.Mxcurr
    currx = @view allcurrx[ix]
    M = @view args.Mat[:,ix]
    y = args.y
    #println(ix,w)
    noiseprop[ix] = pi*(logsigmoid.(w)) .- pi/2
    G[ix] =  1 .- 2*logsigmoid.(w)
    q = similar(w)

    try
        q .= @. (scale*pi*exp(w)*(tan((pi*(exp(w) - 1))/(2*(exp(w) + 1)))^2 + 1))/(exp(w) + 1)^2

    catch
        println(w)
        error("")
    end
    
    compx = scale*tan.(noiseprop[ix])-currx
    Mxprop .= Mxcurr
    Mxprop .+= M*compx

    res .= Mxprop 
    res .-=  y

    G[ix] = G[ix] - diagm(q)/noisevar*M'*(res)
    if(both)
        logp = sum(w-2*log1pexp.(w))
        logp = logp -0.5/noisevar*dotitself(res)
    end

    return logp,G[ix]
 
end


function  logpdiff(U,args,cache)
    M = args.Mat;
    noisesigma = args.noisesigma
    scale = args.scale; y  = args.y; F = args.F

    Fxprop = cache.Fxprop
    Dxprop = cache.Dxprop
    res = cache.residual
    Fxprop .= F*U;
    Dxprop .= M*U;

    res .= Fxprop-y

    logp = -0.5/noisesigma^2*dotitself(res);
    logp = logp + sum(log.(scale./((scale^2 .+ Dxprop.^2))));

    return logp

end

function  logpdiffgradi(U,arguments,cache;both=true)
    M = arguments.Mat; 
    noisesigma = arguments.noisesigma
    scale = arguments.scale; y  = arguments.y; F = arguments.F

    Fxprop = cache.Fxprop
    Dxprop = cache.Dxprop
    res = cache.residual
    Fxprop .= F*U;
    Dxprop .= M*U;

    res .= Fxprop-y
    logp = -0.5/noisesigma^2*dotitself(res);
    logp = logp + sum(log.(scale./((scale^2 .+ Dxprop.^2))));

    #nt = length(y); N = length(U);
    #for i = 1:nt
            #println(Gr[i,:]')
         #q =  (-((r[i]-y[i])./noisesigma.^2))*Gr[i,:]';
         #G = G + dropdims(q, dims = tuple(findall(size(q) .== 1)...))
     #end
     G= F'*(-((res)./noisesigma.^2))

    Gd =   M'*(-2.0*Dxprop./(scale^2 .+ Dxprop.^2));
    #Gd = Gd'*M; 
    G = G + Gd;

    return logp,G

end

function  logpdiffpartial(ix,U,args,cache,current)
    D = @view args.Mat[:,ix]; 
    noisesigma = args.noisesigma
    scale = args.scale; y  = args.y; 
    F  = @view args.F[:,ix]

    logp = 0.0

    currx = @view current.xcurr[ix]
    Fxcurr = current.Fxcurr
    Dxcurr =  current.Dxcurr

    xprop = cache.xprop
    xprop[ix] = U

    compx = U-currx
    Fxprop = cache.Fxprop
    Fxprop .= Fxcurr
    Fxprop .+= F*compx

    Dxprop = cache.Dxprop
    Dxprop .= Dxcurr
    Dxprop .+= D*compx

    res = cache.residual
    res .= Fxprop
    res .-=  y

    logp = -0.5/noisesigma^2*dotitself(res);
    logp = logp + sum(log.(scale./((scale^2 .+ Dxprop.^2))));


    return logp

end

function  logpdiffpartialgradi(ix,U,arguments,cache,current;both=true)

    D = @view arguments.Mat[:,ix]; 
    noisesigma = arguments.noisesigma
    scale = arguments.scale; y  = arguments.y; F = @view arguments.F[:,ix]

    #(Fxcurr=(F*X0),Dxcurr=(Md1*X0), xcurr = copy(X0))

    currx = @view current.xcurr[ix]
    Fxcurr = current.Fxcurr
    Dxcurr =  current.Dxcurr

    xprop = cache.xprop
    xprop[ix] = U

    compx = U-currx  
    Fxprop = cache.Fxprop
    Fxprop .= Fxcurr
    Fxprop .+= F*compx

    Dxprop = cache.Dxprop
    Dxprop .= Dxcurr
    Dxprop .+= D*compx

    res = cache.residual
    res .= Fxprop
    res .-=  y

    logp = 0.0

    if (both)
        logp = -0.5/noisesigma^2*dotitself(res);
        logp = logp + sum(log.(scale./((scale^2 .+ Dxprop.^2))));
    end

    G = cache.gradiprop

    G[ix] = -F'*res/noisesigma^2

    G[ix] = G[ix] + D'*(-2.0*Dxprop./(scale^2 .+ Dxprop.^2));

    return logp,G[ix]

end

function refreshcurrent!(cache,current;mode=1)
    if (mode ==1)
        current.Mxcurr .= cache.Mxprop
        current.noisecurr .= cache.noiseprop
    elseif (mode==2)
        current.xcurr .= cache.xprop
        current.Fxcurr .= cache.Fxprop
        current.Dxcurr .= cache.Dxprop
    end
end

function repatpartial(lpdf,placeholder,ix,x0,args,cache,current,tuning;mode=1,Np=1,Nruns=1,T=-Inf,extra=copy(x0))::Vector{Float64}
    x = copy(x0[ix])
    dim = length(x)

    logpdf(xp) = lpdf(ix,xp,args,cache,current)
    #logpdf(xp) = lpdf(partialreplace(x0,xp,ix),args,cache)
    density = logpdf(x)

    z = copy(extra[ix])
    u = randn(dim)
    densityz = logpdf(z)

    for i = 1:Nruns
        R = tuning.Cho.L/sqrt(2)
		u .= randn(dim)
		xtilde = x + R * u
		densitytilde = logpdf(xtilde)
		while (log(rand()) >=  density - densitytilde)
			u .= randn(dim)
			xtilde .= x + R * u
			densitytilde = logpdf(xtilde)
		end

		u .= randn(dim)
		xstar = xtilde + R * u
		densitystar = logpdf(xstar)

		while (log(rand()) >=  densitystar - densitytilde)
			u .= randn(dim)
			xstar .= x + R * u
			densitystar = logpdf(xstar)
		end

		u .= randn(dim)
		zstar = xstar + R * u
		densityzstar = logpdf(zstar)

		while (log(rand()) >=  densitystar - densityzstar)
			u .= randn(dim)
			zstar .= xstar + R * u
			densityzstar = logpdf(zstar)
		end

        ratio = densitystar + min(0,density-densityz) - density - min(0,densitystar-densityzstar)
        if (log(rand()) <= ratio)
            x .= xstar
            z .= zstar
			extra[ix] = zstar
			density = densitystar
			densityz = densityzstar
            
        end
        tuning.n = tuning.n + 1

        if (tuning.n <= tuning.Nadapt)
            xmp = copy(tuning.xm)
            tuning.xm .= tuning.xm + (x-tuning.xm)/tuning.n
            xm = tuning.xm
            tuning.C .= ((tuning.n-1)*tuning.C + (x-xm)*(x-xmp)')/tuning.n
            tuning.Cho = cholesky(Hermitian(2.38^2/dim*tuning.C + 1e-11*I(dim)))
        end

        
    end
    logpdf(x)
    refreshcurrent!(cache,current;mode=mode)
    return x
end

function sfhmcpartial(lpdf,gradilpdf,ix,x0,args,cache,current,tuning;Nruns=1,mode=1,extra=nothing,Np=-Inf,T=-Inf)
    x = copy(x0[ix])
    Np = length(x)

    logpdf(xp) = lpdf(ix,xp,args,cache,current)
    #logpdf(xp) = lpdf(partialreplace(x0,xp,ix),args,cache)

    glpdf(xp) = gradilpdf(ix,xp,args,cache,current;both=false)[2]
    #glpdf(xp) = gradilpdf(partialreplace(x0,xp,ix),args,cache;both=false)
  
    L = 30
    xo = copy(x)
    xp = copy(xo)
    p = similar(xp)
    alpha = 1.5
    ca = gamma(alpha+1)/gamma(alpha/2+1)^2
    gam = 0.0008
    eta = 0.005

    epsilon = rand(AlphaStable(α=alpha),Np)

    for i=1:Nruns
        p .= randn(Np)
        for j = 1:L
            epsilon .= rand(AlphaStable(α=alpha),Np)
            xp .= xo +ca*eta*p
            p .= (1-eta*gam)*p + ca*eta*glpdf(xp) + (gam*eta)^(1/alpha)*epsilon
            xo .= xp
        end
        #x .= xp
    end

    logpdf(xp)
    refreshcurrent!(cache,current;mode=mode)

    return xp
end

function totallogpdf(x,target,N,T)
    Ncompo = Int64(length(x)/N)
    q = Array{Float64}(undef, N)
    w = 0.0
    for i = 1:N
        p = @view x[(i-1)*Ncompo+1:i*Ncompo]
        v = target(p)
        if (isnan(v))
            println(x)
            @assert !isnan(v)
        end
        q[i] = v - v/T
        w = w + v/T
    end

    w = w + logsumexp(q)
    
    return w

end

function totalloggrad(x,target,targetgrad,N,T)
    J = zeros(length(x),)
    Ncompo = Int64(length(x)/N)
    q = Array{Float64}(undef, N)
    gradis = zeros(Ncompo,N)
    w = 0.0
    for i = 1:N
        p = @view x[(i-1)*Ncompo+1:i*Ncompo]
        v = target(p)
        q[i] = v - v/T
        w = w + v/T
        gradis[:,i] = targetgrad(p)
        J[(i-1)*Ncompo+1:i*Ncompo] += gradis[:,i]/T
    end

    w = w + logsumexp(q)
    q .= softmax(q)
    #

    for i = 1:N
        J[(i-1)*Ncompo+1:i*Ncompo] += q[i]*(gradis[:,i]*(1-1/T))
    end

    return w,J

end


function reweight(x,Npseudo,target,T)
    #Ns = size(samples)[2]
    Ncompo = Int64(size(x)[1]/Npseudo)
    R = zeros(Ncompo,)

    iv = Vector(1:Npseudo)
    #for i = 1:Ns
    w = zeros(Npseudo,)
    for j = 1:Npseudo
        w[j] = target(x[(j-1)*Ncompo+1:j*Ncompo])*(1-1/T)
    end
    w .= exp.(w .- maximum(w))
    index = sample(iv,StatsBase.pweights(w))
    R = x[(index-1)*Ncompo+1:index*Ncompo]
    #end

    return R
end

function pehmcpartial(lpdf,gradilpdf,ix,x00,args,cache,current,tuning;Nruns=5,Np=3,T=5.0,mode=1,extra=nothing)
    x0 = repeat(x00[ix],Np)
    x = copy(x0)
    Nall = length(x)

    xnew = similar(x)
    pnew = similar(xnew)
    p = similar(xnew)

    logpdf_can(xp) = lpdf(ix,xp,args,cache,current)
    #logpdf_can(xp) = lpdf(partialreplace(x00,xp,ix),args,cache)

    logpdf(x) = totallogpdf(x,logpdf_can,Np,T)  

    glpdf_can(xp) = gradilpdf(ix,xp,args,cache,current;both=false)[2]
    #glpdf_can(xp) = gradilpdf(partialreplace(x00,xp,ix),args,cache)[2][ix]

    glpdf(x) = totalloggrad(x,logpdf_can,glpdf_can,Np,T)[2]
    glpdfboth(x) = totalloggrad(x,logpdf_can,glpdf_can,Np,T)

    # if (Nruns == 0.0)
    #     if (tuning.n < tuning.Nadapt)
    #         metric = UnitEuclideanMetric(Nall)
    #         hamiltonian = Hamiltonian(metric, glpdf, glpdfboth)   
    #         epsilooni = find_good_stepsize(hamiltonian,x;max_n_iters=15)
    #         integrator =  Leapfrog(epsilooni)     
    #         proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator,9,1000.0)
    #         xp, _ = sample(hamiltonian, proposal, x, 1; verbose=false, progress=false)
    #         x .= xp[1]
      
    #     elseif (tuning.n <= tuning.Nadapt +2)
    #         metric = UnitEuclideanMetric(Nall)
    #         hamiltonian = Hamiltonian(metric, glpdf, glpdfboth)   
    #         epsilooni = find_good_stepsize(hamiltonian,x;max_n_iters=20)
    #         integrator =  Leapfrog(epsilooni)     
    #         tuning.adaptori = StepSizeAdaptor(0.85, integrator) 
    #         proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator,9,1000.0)
    #         xp,_  = sample(hamiltonian, proposal, x, 30, tuning.adaptori, 30;verbose=false, progress=false)
    #         x .= xp[1] 
    #         println(tuning.adaptori.state.ϵ)

    #     else
    #         metric = UnitEuclideanMetric(Nall)
    #         hamiltonian = Hamiltonian(metric, glpdf, glpdfboth)
    #         integrator =  Leapfrog(tuning.adaptori.state.ϵ)
    #         proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator,9,1000.0)    
    #         xp, _ = sample(hamiltonian, proposal, x, 1; verbose=false, progress=false)
    #         x .= xp[1]
        
    #     end

    # else
    #     metric = UnitEuclideanMetric(Nall)
    #     hamiltonian = Hamiltonian(metric, glpdf, glpdfboth)   
    #     integrator =  Leapfrog(Nruns)     
    #     proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator,9,1000.0)
    #     xp, _ = sample(hamiltonian, proposal, x, 1; verbose=false, progress=false)
    #     x .= xp[1]
    # end
    
    # L = 20
    # R = 1.0
    # jitter = 0.2
    # Linc = floor(Int,jitter*L)
    # mu = tuning.mu
    # gamma = 0.05
    # t0 = 10.0
    # kappa = 0.75
    # delta = 0.7

    ll = logpdf(x)   
  
    for _ = 1:Nruns

        tuning.n = tuning.n + 1

        
        # p .= R*randn(Nall)
        # #leapfrog_m!(xnew,pnew,x,p,Gi,IG,eG,glpdf,epsilon,L+rand(-Linc:Linc))
        # Lnow = L + rand(-Linc:Linc)
        # epsilonnow = tuning.epsilon# + rand()
        # leapfrog!(xnew,pnew,x,p,1.0,glpdf,epsilonnow,Lnow)
        # lnew = logpdf(xnew)

        # U = rand()
        # apr = min(1.0,exp(-ll + lnew - 1/2*pnew'*pnew +1/2*p'*p))
        
        

        # if (tuning.n<tuning.Nadapt)
        #     tuning.H = (1-1/(t0+tuning.n))*tuning.H + 1/(t0+tuning.n)*(delta-apr)
        #     tuning.epsilon = exp(mu - sqrt(tuning.n)/gamma*tuning.H)
        #     tuning.epsilon_hat = exp(tuning.n^(-kappa)*log(tuning.epsilon) + (1-tuning.n^(-kappa))*log(tuning.epsilon_hat))
        # else
        #     tuning.epsilon = tuning.epsilon_hat
            
        # end

        V=randn(Nall)
        xnew .= x+ tuning.Cho.L*V

        lnew = logpdf(xnew)

        U = rand()
        apr = min(1.0,exp(-ll + lnew))
        
        if (U < apr)
            ll = lnew
            x .= xnew
            tuning.acc = tuning.acc +1    
        end


        if (tuning.n <= tuning.Nadapt)
            xmp = copy(tuning.xm)
            tuning.xm .= tuning.xm + (x-tuning.xm)/tuning.n
            xm = tuning.xm
            tuning.C .= ((tuning.n-1)*tuning.C + (x-xm)*(x-xmp)')/tuning.n
            tuning.Cho = cholesky(Hermitian(2.38^2/Nall*tuning.C + 1e-11*I(Nall)))

            # atarg = 0.3
            # V = V/norm(V)
            # z = sqrt(tuning.n^(-0.7) *abs(tuning.acc/tuning.n - atarg)) *  tuning.Cho.L*V
            # #println(z)
            # if (tuning.acc/tuning.n >= atarg)
            #     lowrankupdate!(tuning.Cho, z)
                
            # else
            #     lowrankdowndate!(tuning.Cho, z)
            # end


        end

    end
    
    xret = reweight(x,Np,logpdf_can,T)
    logpdf_can(xret)
    refreshcurrent!(cache,current;mode=mode)

    return xret

end

function mtmpartial(lpdf,gradilpdf,ix,x00,args,cache,current,tuning;mode=1,Np=10,Niter=2,extra=nothing,T=-Inf)

    x0 =  copy(x00[ix])
    Nall = length(x0)

    # metric = UnitEuclideanMetric(Nall)
    # integrator =  Leapfrog(0.01)

    logpdf =  Vector{Function}(undef,Np)
    # glpdf = Vector{Function}(undef,Np)
    # glpdfboth = Vector{Function}(undef,Np)
    # hamiltonian = Vector{Hamiltonian}(undef,Np)
    # proposal = Vector{NUTS}(undef,Np)

    for i = 1:Np
        logpdf[i] = xp-> lpdf(ix,xp,args,cache[i],current)
        # glpdfboth[i] = xp->  gradilpdf(ix,xp,args,cache[i],current;both=true)
        # glpdf[i] = xp->  gradilpdf(ix,xp,args,cache[i],current;both=false)
        # hamiltonian[i] = Hamiltonian(metric, glpdf[i], glpdfboth[i])
        # proposal[i] = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator,11,1000.0) 
    end

    #logpdf(xp) =  lpdf(ix,xp,args,cache,current)
    #logpdf(xp) = lpdf(partialreplace(x00,xp,ix),args,cache)

    #glpdf(xp) = gradilpdf(xp)# gradilpdf(ix,xp,args,cache,current;both=false)[2]
    #glpdf(xp) = gradilpdf(partialreplace(x00,xp,ix),args,cache)[2][ix]
    #glpdfboth(xp) = (logpdf(xp),glpdf(xp))


    # if (tuning.n>tuning.Nadapt)
    #     metric = UnitEuclideanMetric(Nall)
    #     hamiltonian = Hamiltonian(metric, glpdf, glpdfboth)
    #     integrator =  Leapfrog(tuning.adaptori.state.ϵ)
    #     proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator,6,1000.0)    
    # else
    #     metric = UnitEuclideanMetric(Nall)
    #     hamiltonian = Hamiltonian(metric, glpdf, glpdfboth)
    #     if (isnothing(tuning.adaptori))      
    #         epsilooni = find_good_stepsize(hamiltonian,x;max_n_iters=10)
    #         integrator =  Leapfrog(epsilooni)
    #         tuning.adaptori = StepSizeAdaptor(0.8, integrator) 
    #     end       
    #     integrator =  Leapfrog(tuning.adaptori.state.ϵ)
    #     proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator,6,1000.0)      
    # end

    s2 = 0.1
    xpr = copy(x0)
    lwn = zeros(Np,)
    llwn = zeros(Np,)
    kwn = zeros(Np,)

    #println(find_good_stepsize(hamiltonian[1],x0;max_n_iters=10))

    plpdf(x,xa) = -0.5/s2*dotitself(x-xa)
    x = [zeros(Nall) for _ = 1:(Np)]

    for k = 1:Niter 
        #  for i = 1:Np+1
        #     if(k == 1)
        #         lwn[i] = logpdf[i](x[i]) - plpdf(x[i])
        #     else
        #         lwn[i] = 1.0
        #     end
        # end

        Threads.@threads  for i = 1:Np
            x[i] .= sqrt(s2)*randn(Nall) + xpr
            lwn[i] = logpdf[i](x[i]) - plpdf(x[i],xpr)
        end
        

        llwn .= exp.(lwn .- maximum(lwn));
        index = sample(1:Np,StatsBase.pweights(llwn),1)[1]
        xcand = copy(x[index])

        Threads.@threads  for i = 1:Np
            x[i] .= sqrt(s2)*randn(Nall) + xcand   
            kwn[i] = logpdf[i](x[i]) - plpdf(x[i],xcand)
        end

        u = log(rand())
        if (u <= logsumexp(lwn)-logsumexp(kwn))
            xpr .= xcand
        end




        
        # Threads.@threads  for i = 1:Np
        #     p, _ = sample(hamiltonian[i], proposal[i],  x[i], 1; verbose=false, progress=false)
        #     x[i] = p[1]
        # end
        # println(cache[1].xprop[ix])
        # println(x)
        # error("")
    end
    
    #x,_  = sample(hamiltonian, proposal, x, 10, tuning.adaptori, 10;verbose=false, progress=false)

    # println(cache[1].xprop[ix])
    #  for i = 1:Np
    #     #x[i] = x[1]
    #     lwn[i] = logpdf[i]( x[i])
    # end
    # println(cache[1].xprop[ix])


    # lwn .= exp.(lwn .- maximum(lwn));
    # lwn .= lwn./sum(lwn)
    # index = sample(1:Np,StatsBase.pweights(lwn),1)[1]
    #xret =  xpr
    #println(lwn)
    #error("")

    #tuning.n = tuning.n + 1
    logpdf[1](xpr)
    refreshcurrent!(cache[1],current;mode=mode)
    
    # println(cache[index].xprop[ix],current.xcurr[ix])
    # error("")

    return xpr

end

mutable struct rram    
    n::Int64
    acc::Int64
    Nadapt::Int64
    C::Array{Float64,2}
    Cho::Cholesky{Float64,Array{Float64,2}}
    xm::Vector{Float64}
end

mutable struct ahmc
    n::Int64
    adaptori::Union{Nothing,Int64,NesterovDualAveraging{Float64}}
    Nadapt::Int64
end

mutable struct dahmc2
    epsilon::Float64
    epsilon_hat::Float64
    mu::Float64
    H::Float64
    Nadapt::Int64
    n::Int64
    acc::Int64
end

function totalsample(lpdf,glpdf,x0,meas,arg,ix;N=1,Nadapt=1000,Nruns=5,thinning=10,algo=repatpartial,mode=1,Np=2,T=2.0)
    x = copy(x0)
    Npar = length(x0)
    chain = zeros(Npar, Int(floor(N / thinning)))
    Nix = length(ix)   

    if (mode==1)
        noisecurr = arg.scale*tan.(pi*(logsigmoid.(x)) .- pi/2)
        Mxcurr = arg.Mat*noisecurr
        current = (Mxcurr = Mxcurr, noisecurr = noisecurr)
        cache=(noiseprop=copy(noisecurr),Mxprop=copy(Mxcurr),residual=similar(meas),gradiprop=similar(x0))
    elseif (mode==2)
        cache=(xprop=copy(x0),Fxprop=(arg.F*x0),Dxprop=(arg.Mat*x0), residual=similar(meas),gradiprop=similar(x0))
        current = (Fxcurr=(arg.F*x0),Dxcurr=(arg.Mat*x0), xcurr = copy(x0))   
    else
        cache = []
        current = []
    end

    if algo == repatpartial
        extra = copy(x0)
        tuning = Vector{rram}(undef,Nix)
        for k = 1:Nix
            n = length(ix[k])
            C = 0.1*Array(I(n))
            Cho = cholesky(C)
            xm = copy(x0[ix[k]])
            tuning[k] = rram(1,0,Nadapt,C,Cho,xm)
        end

    elseif (algo==pehmcpartial)
        extra = []
        tuning = Vector{rram}(undef,Nix)
        #tuning = Vector{dahmc2}(undef,Nix)
        #tuning = Vector{ahmc}(undef,Nix)
        for k = 1:Nix

            n = length(ix[k])*Np
            C = 5*Array(I(n))
            Cho = cholesky(C)
            xm = repeat(copy(x0[ix[k]]),Np) 
            tuning[k] = rram(1,0,Nadapt,C,Cho,xm)

            # e0 = 0.01; mu0 = log(10*e0)
            # tuning[k]=dahmc2(e0,1.0,mu0,0.0,Nadapt,0,0)
            #tuning[k] = ahmc(0,nothing,Nadapt)
        end

    elseif (algo==mtmpartial)
        tuning = Vector(1:Nix)
        extra = []
        cache2 = deepcopy(cache)
        cache = Vector{typeof(cache2)}(undef,Np+1)
        for i = 1:Np+1
            cache[i] = deepcopy(cache2)
        end


    elseif (algo==sfhmcpartial)
        tuning = Vector(1:Nix)
        extra = []
    end


    pb = ProgressBar(1:N)

    for i in pb
        for k = 1:Nix
            x[ix[k]] = algo(lpdf,glpdf,ix[k],x,arg,cache,current,tuning[k];mode=mode,Nruns=Nruns,extra=extra,Np=Np,T=T)#lpdf,ix,x0,args,cache,current,tuning
            if (i % thinning == 0)
                chain[ix[k], Int(i / thinning)] = x[ix[k]]
            end
           
        end
    end

    return chain,tuning
end

function savechain(c,prefix)
    p = pwd()
    f = p*"/chains/"*prefix*"_"*Dates.format(Dates.now(), "yyyy_mm_dd_HH_MM_SS")*".mat"
    matwrite(f,Dict("chain"=>c))

end


N = 200;
x = Vector(range(-0.0,stop=1.0,length=N));    dx = x[2]-x[1];
xi =  copy(x)
measxi = xi ;


triangle(x) = 1.0*(x+1)*(x<=0)*(x>=-1) + 1.0*(-x+1)*(x>0)*(x<=1);
heavi(x) = 1.0*(x>=0);
tf(x) = 0+ 0.2*heavi(x-0.1)*heavi(-x+0.15) - 0.5*heavi(x-0.65)*heavi(-x+0.7)+ 0*exp(-60*abs(x-0.2)) - 0*exp(-180*abs(x-0.8)) + 0*triangle(10*(x-0.8)) + 1*heavi(x-0.3)*heavi(-x+0.6);
#tf(x) = triangle(30*(x-0.8)) + exp(-90*abs(x-0.2)) + heavi(x-0.45)*heavi(-x+0.55)
sigma = 0.15;  # Standard deviation of the measurement noise. #Cauchy1D = 0.15. ¤SPDE: 0.05
cw = 100;
kernel(x,y) =   cw/2*exp(-cw*abs((x-y)));
tfc(x) = quadgk(y ->  tf(y)*kernel(x,y), 0, 1, rtol=1e-7)[1]

F = measurementmatrix(xi,measxi,kernel); # Theory matrix.
gt = tf.(xi);
gtc =  tfc.(measxi); #F*gt

Random.seed!(100)

meas = gtc .+ sigma.*randn(size(measxi)); # Simulate the measurements.

lcauchy = 0.003;
acauchy = lcauchy^2; bcauchy = 1;
cauchypriorscale = 0.005;

cd1scale = 0.03;
cd2scale = 0.01;

Md1=difference1(xi);
Md1i=Md1\I(N);

Mcauchy = smatrix(x,acauchy,bcauchy); MN = size(Mcauchy)[1];
Micauchy = Mcauchy\I(MN);

measnoisescaling = 1.0
diff1arg = (F=F, y=meas,Mat=sparse(Md1),scale=cd1scale,noisesigma=measnoisescaling*sigma)
cauchyarg = (Mat=Micauchy,y=meas,scale=cauchypriorscale,noisesigma=measnoisescaling*sigma)
cauchyreparg = (Mat=F*Micauchy,y=meas,scale=cauchypriorscale,noisesigma=measnoisescaling*sigma)

cached=(xprop=similar(meas),Fxprop=(F*meas),Dxprop=(Md1*meas), residual=similar(meas),gradiprop=similar(meas))
cachem=(noiseprop=similar(xi),Mxprop=similar(xi),residual=similar(meas),gradiprop=similar(xi))

# targetdiff(w) = -logpdiff(w,diff1arg,cached)#-logpcauchy(w,cauchyarg)# 
# targetdiffgrad(w) = -logpdiffgradi(w,diff1arg,cached;both=false)[2]#-logpcauchygradi(w,cauchyarg)[2]# 
# res = Optim.optimize(targetdiff, targetdiffgrad,randn(N,),Optim.LBFGS(), Optim.Options(allow_f_increases=true,show_trace=false,iterations=1000); inplace = false)
# MAP_diff = res.minimizer



# close("all")
# plot(gt)
# plot(meas)
# plot(MAP_diff)

# Ntot = length(xi)

#error("")

function NUTS_sample(N,arg,cac)
    n_samples, n_adapts = 400, 300
    ss = Vector{Any}(undef,N)
    st = Vector{Any}(undef,N)
    Np = 1
    T = 3.0
    for i = 1:N

        Ntot = length(xi)*Np
        Random.seed!(i)
        initial_θ = repeat(0.1*randn(length(xi)),Np)
        global c

        # lpn(w) = logpdiff(w,arg,cac)
        # targetdiff(x) = totallogpdf(x,lpn,Np,T)  
        # gpn(w) =  logpdiffgradi(w,arg,cac;both=false)[2]
        # targetdiffgrad(x) = totalloggrad(x,lpn,gpn,Np,T)

          #targetdiff(w) = logpdiff(w,arg,cac)
        #targetdiffgrad(w) = logpdiffgradi(w,arg,cac;both=true)

        targetdiff(w) = logpcauchyrep(w,arg,cac)
        targetdiffgrad(w) = logpcauchyrepgradi(w,arg,cac;both=true)

        metric = DiagEuclideanMetric(Ntot)# DenseEuclideanMetric(Ntot)# DiagEuclideanMetric(Ntot)
        hamiltonian = Hamiltonian(metric, targetdiff, targetdiffgrad)

        initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
        integrator =  Leapfrog(initial_ϵ)#  TemperedLeapfrog(initial_ϵ, 1.05)# Leapfrog(initial_ϵ)
        proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator,12,1000.0)
        adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
        c, stats = sample(hamiltonian, proposal, initial_θ, n_samples, adaptor, n_adapts; progress=true)
        ss[i] = hcat(c...)
        # c = reweight_chain(c,Np,lpn,T)
        # ss[i] = c
        st[i] = stats
        pr="NUTS_"*string("spde_normal")
        savechain(c,pr)

    end
    return ss, st
end
c,b = NUTS_sample(0,cauchyreparg,cachem)


# Mgauss = smatrix(x,agauss,bgauss); MN = size(Mgauss)[1];
# Migauss = Mgauss\I(MN);
# Migauss=Migauss^((nu+0.5)/2); Migauss = real(Migauss);
# lgauss = 0.08; scalestd = 1; nu = 3/2;
# agauss = lgauss^2; bgauss = 1;
# alphagauss = scalestd^2*2*pi^0.5*gamma(nu+1/2)/gamma(nu);
#gausspriorsigma = sqrt(0.5*alphagauss*lgauss/dx);
#theoreticalcovariance(x) = scalestd^2*2^(1-nu)/gamma(nu)*(abs(x)/lgauss)^nu*besselk(nu,abs(x)/lgauss);



#cauchyarg = (F=F, y=meas,Mi=Micauchy,scale=cauchypriorscale,noisesigma=measnoisescaling*sigma)
# cauchyreparg = (Mat=F*Micauchy,y=meas,scale=cauchypriorscale,noisesigma=measnoisescaling*sigma)

#diff1arg = (F=F, y=meas,Mat=sparse(Md1),scale=cd1scale,noisesigma=measnoisescaling*sigma)
#diff1reparg = (Mat=F*Md1i,y=meas,scale=cd1scale,noisesigma=measnoisescaling*sigma)

#W = randn(N,)
#W2 = copy(W); 
#WC = ComplexF64.(copy(W))
#W2C =  ComplexF64.(copy(W2))

#noisecurr = diff1reparg.scale*tan.(pi*(logsigmoid.(W)) .- pi/2)
#Mxcurr = diff1reparg.Mat*noisecurr

#cache=(noiseprop=copy(noisecurr),Mxprop=copy(Mxcurr),residual=similar(meas),gradiprop=similar(noisecurr))
#cachec=(noiseprop=ComplexF64.(cache.noiseprop),Mxprop=ComplexF64.(cache.Mxprop),residual=ComplexF64.(cache.residual),gradiprop=similar(noisecurr))

# funktio = logpcauchyrep
# grfunktio = logpcauchyrepgradi

# funktio = logpcauchyreppartial
# grfunktio = logpcauchyreppartialgradi

# funktio = logpdiffpartial
# grfunktio = logpdiffpartialgradi

#funktio = logpdiff
#grfunktio = logpdiffgradi

#f(x) = log(1/(0.001+(x[1]-2)^2)*1/(0.001+(x[1]+2)^2))
# f(x) = logsumexp([(-0.5/0.01* (x[1]-1)^2) , (-0.5/0.01*(x[1]+1)^2) ])
# # #f(x) = -0.5*(x[1]-2)^2
# g(x) = ForwardDiff.gradient(f,x)

ix = divide(N,1)
algo =  repatpartial#pehmcpartial
grfunk = logpcauchyreppartialgradi# logpdiffpartialgradi# 
funk =  logpdiffpartial#logpcauchyreppartial#logpdiffpartial#
par = cauchyreparg#diff1arg# 
mode = 1# 1 
x000  = copy(MAP_diff) #randn(N,)

cached=(xprop=copy(x000),Fxprop=(F*x000),Dxprop=(Md1*x000), residual=copy(meas),gradiprop=copy(gt))
currentd = (Fxcurr=(F*x000),Dxcurr=(Md1*x000), xcurr = copy(x000))
noisecurrm = cauchyreparg.scale*tan.(pi*(logsigmoid.(x)) .- pi/2)
Mxcurrm = cauchyreparg.Mat*noisecurrm
currentm = (Mxcurr = Mxcurrm, noisecurr = noisecurrm)
cachem=(noiseprop=copy(noisecurrm),Mxprop=copy(Mxcurrm),residual=similar(meas),gradiprop=similar(x000))

# funkt(ix,x,a,cc,ca) = f(x)
# grfunkt(ix,x,a,cc,ca;both=true) = f(x),g(x)
# ix = [[1],]

xq = Vector(-3:0.01:3)
q = map( v->  funk([60],[v],diff1arg,cached,currentd), xq )
q = exp.(q .- maximum(q))
plot(xq,q)
 error("")

#Kokeile: Caichu-deiff itsessään kolmella menetelmällä.

#  mode = 4

for t=1:0
    Random.seed!(t)
    xz = 0.1*randn(N,)

    Nsamples = 400; Nadapt = 200; thinning = 1
    Np = 2; Nruns = 10; T = 50
    c,tun=totalsample(funk,grfunk,xz,meas,par,ix;N=Nsamples,Nadapt=Nadapt,thinning=thinning,algo=algo,mode=mode,Np=Np,T=T,Nruns=Nruns)
    savechain(c,string(nameof(algo))*"_"*string(nameof(funk))*"_N"*string(Nsamples)*"_Nadapt"*string(Nadapt)*"_thin"*string(thinning)*"_Np"*string(Np)*"_Nruns"*string(Nruns)*"_T"*string(T))
    println(effective_sample_size(c[1,:]))
    global c
    global tun
end


 cmapper(x) = cauchyreparg.scale*tan.(pi*(logsigmoid.(x)) .-pi/2)
# # if (mode==1)
     mapper(x) = Micauchy*cmapper(x)

# # else
# #     mapper(x) = x
# # end
# pp = similar(c)
# for i = 1:size(c)[2]
#      pp[:,i] = mapper(c[:,i])
#  end


#pix = [2,3]

# X0 = copy(meas)
# X0C = ComplexF64.(X0)
# xx = copy(X0[pix])

# # g(x) = logpdiff(partialreplace(X0,x,pix),diff1arg)
# # a1=gradic(g,xx;eps=1e-12)
# # a2 = logpdiffgradi(X0,diff1arg)[2][pix]

# g(x) = logpdiffpartial(pix,x,diff1arg,cached,currentd)
# g(x) = logpdiff(partialreplace(X0C,x,pix),diff1arg)
# a1 = gradic(g,xx;eps=1e-12)
# a2 = logpdiffpartialgradi(pix,xx,diff1arg,cached,currentd)[2]

# # _, a3 = logpcauchyrepgradi(partialreplace(W,xx,pix),diff1reparg,cache)
# # a3 = a3[pix]

# # _,a2=logpcauchyreppartialgradi(pix,xx,diff1reparg,cache,current;both=false)

# println(a1-a2)

# println(logpcauchyreppartial(ix,view(W,ix),diff1reparg,cache,current) -logpcauchyreppartial(ix,view(W2,ix),diff1reparg,cache,current))
# refreshcurrent!(cache,current)
# println(logpcauchyreppartial(ix,view(W,ix),diff1reparg,cache,current) -logpcauchyreppartial(ix,view(W2,ix),diff1reparg,cache,current))


#TODO: ARMS?
# Muistutus:  SFHMC kaikkoi kompnentit, PE-hmC ja RAm vain osa. NUTS kaikki. 
