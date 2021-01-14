using LinearAlgebra
using Random
using PyPlot
using Interpolations
using SparseArrays
using Optim

include("apu.jl")

Random.seed!(1)

function tt(x,args)
    noisesigma = args.noisesigma
    scale = args.scale; y  = args.y; F = args.F
    bscale = args.bscale
    Dx = 0*args.Dx; Dy = args.Dy
    Db = args.Db

    res = F*x - y
    Lxx = Dx*x
    Lyx = Dy*x


    G = 2*(Dx'*(Lxx./(Lxx.^2 + Lyx.^2))) .+  2*(Dy'*(Lyx./(Lxx.^2 + Lyx.^2)))

    return sum(log.(Lyx.^2 + Lxx.^2)),G

end

function  logpdiff(x,args,cache)
    D = args.D;
    noisesigma = args.noisesigma
    scale = args.scale; y  = args.y; F = args.F

    Fxprop = cache.Fxprop
    Dprop = cache.Dprop
    res = cache.residual
    Fxprop .= F*x;
    Dprop .= D*x;

    res .= Fxprop-y

    logp = -0.5/noisesigma^2*dotitself(res);
    logp = logp + sum(log.(scale./((scale^2 .+ Dprop.^2))));

    return logp

end

function  logpdiffgradi(x,arguments,cache;both=true)
    D = arguments.D; 
    noisesigma = arguments.noisesigma
    scale = arguments.scale; y  = arguments.y; F = arguments.F

    Fxprop = cache.Fxprop
    Dprop = cache.Dprop
    res = cache.residual
    Fxprop .= F*x;
    Dprop .= D*x;

    res .= Fxprop-y
    logp = -0.5/noisesigma^2*dotitself(res);
    logp = logp + sum(log.(scale./((scale^2 .+ Dprop.^2))));

    #nt = length(y); N = length(U);
    #for i = 1:nt
            #println(Gr[i,:]')
         #q =  (-((r[i]-y[i])./noisesigma.^2))*Gr[i,:]';
         #G = G + dropdims(q, dims = tuple(findall(size(q) .== 1)...))
     #end
     G = F'*(-((res)./noisesigma.^2))

    Gd = D'*(-2.0*Dprop./(scale^2 .+ Dprop.^2));
    #Gd = Gd'*M; 
    G = G + Gd;

    return logp,G

end
function  logpisodiff(x,args,cache)
    noisesigma = args.noisesigma
    scale = args.scale; y  = args.y; F = args.F
    bscale = args.bscale
    Dx = args.Dx; Dy = args.Dy
    Db = args.Db

    res = cache.residual
    Dxprop = cache.Dxprop
    Dyprop = cache.Dyprop
    Dbprop = cache.Dbprop
    Fxprop = cache.Fxprop

    Fxprop .= F*x
    res .= Fxprop - y  
    Dxprop .= Dx*x
    Dyprop .= Dy*x
    Dbprop .= Db*x

    return   -0.5/noisesigma^2*res'*res -3/2*sum(log.(scale^2 .+ Dxprop.^2 .+ Dyprop.^2)) - sum(log.(bscale^2 .+ Dbprop.^2))

end

function logpisodiffgradi(x,args,cache;both=true)
    noisesigma = args.noisesigma
    scale = args.scale; y  = args.y; F = args.F
    bscale = args.bscale
    Dx = args.Dx; Dy = args.Dy
    Db = args.Db

    logp = 0.0

    res = cache.residual
    Dxprop = cache.Dxprop
    Dyprop = cache.Dyprop
    Dbprop = cache.Dbprop
    Fxprop = cache.Fxprop
    G = cache.gradiprop

    Fxprop .= F*x
    res .= Fxprop - y  
    Dxprop .= Dx*x
    Dyprop .= Dy*x
    Dbprop .= Db*x

    G .= F'*(-((res)./noisesigma.^2))

    if both
        logp = -0.5/noisesigma^2*res'*res -3/2*sum(log.(scale^2 .+ Dxprop.^2 .+ Dyprop.^2)) - sum(log.(bscale^2 .+ Dbprop.^2))
    end

    #Gd1 =  Dx'*(-3.0*Lxx./(scale^2 .+ Lxx.^2));
    G .= G  -3.0*(Dx)'*(Dxprop./(scale^2 .+ Dyprop.^2 + Dxprop.^2))  - 3.0*(Dy)'*(Dyprop./(scale^2 .+ Dyprop.^2 + Dxprop.^2))   -Db'*(2.0*Dbprop./(bscale^2 .+ Dbprop.^2));

    return logp, G 
end

function regmatrices_first(dim)
    reg1d = diagm(Pair(0,-1*ones(dim))) + diagm(Pair(1,ones(dim-1))) + diagm(Pair(-dim+1,ones(1))) ;reg1d[dim,dim] = 0
    #reg1d = reg1d[1:dim-1,:]
    iden = I(dim)
    regx = kron(reg1d,iden)
    regy = kron(iden,reg1d)

    rmxix = sum(abs.(regx) ,dims=2) .< 2
    rmyix = sum(abs.(regy) ,dims=2) .< 2
    boundary = ((rmxix + rmyix)[:]) .!= 0
    q = findall(boundary .== 1)
    regx = regx[setdiff(1:dim^2,q), :] 
    regy = regy[setdiff(1:dim^2,q), :] 
    
    s = length(q)
    bmatrix = zeros(s,dim*dim)
    for i=1:s
        v = q[i]
        bmatrix[i,v] = 1
    end
    #bmatrix = bmatrix[i,i] .= 1

    return regx,regy,bmatrix
end

function regmatrices_second(dim)
    reg1d = diagm(Pair(0,2*ones(dim))) + diagm(Pair(1,-1*ones(dim-1))) + diagm(Pair(-1,-1*ones(dim-1))) #;reg1d[dim,dim] = 0
    #reg1d = reg1d[1:dim-1,:]
    iden = I(dim)
    regx = kron(reg1d,iden)
    regy = kron(iden,reg1d)

    rmxix = sum(abs.(regx), dims=2) .< 4
    rmyix = sum(abs.(regy), dims=2) .< 4
    boundary = ((rmxix + rmyix)[:]) .!= 0
    q = findall(boundary .== 1)
    regx = regx[setdiff(1:dim^2,q), :] 
    regy = regy[setdiff(1:dim^2,q), :] 

    #q = findall(boundary .== 1)
    #bmatrix = zeros(dim*dim,dim*dim)
    #for i in q
    #     bmatrix[i,i] = 1
    #end

    h2 = zeros(2,N)
    h2[1,1] = 1; h2[2,N] = 1
    h1 = hcat([zeros(dim-2),I(dim-2),zeros(dim-2)]...)
    bmatrix = [ kron(h2,I(dim)); kron(h1,h2) ]

    return regx,regy,bmatrix
end

function spdematrix(xs,a,b)
    dx = abs(xs[2]-xs[1])
    N = length(xs)
    
    M = zeros(N,N);
    for i=2:N-1
        M[i,i-1] = -a/dx^2;
        M[i,i+1] = -a/dx^2;
        M[i,i] = 2*a/dx^2;# +b;

    end

    M[1,1] = 1 # Option 1
    M[N,N] = 1
    M[N,N-1] = -1 

    #M[1,:] = [2*a/dx^2;-a/dx^2; zeros(N-3);-a/dx^2]; # Option 2
    #M[N,:] = [-a/dx^2; zeros(N-3); -a/dx^2;2*a/dx^2];
    
    M = kron(M,I(N)) + kron(I(N),M)
    M = M + I(size(M,2))*b;

    return M
end       

cw = 150.0
kernel(xi,xj,yi,yj;constant=cw) = constant/pi*exp(-constant*((xi-xj)^2 + (yi-yj)^2) )
tf(x,y) =  15.0*exp.(-20*sqrt.((x .- 0.3).^2 .+ (y .- 0.3).^2)) + 10*((y-x) .< 0.7).*((y-x) .>= -0.7).*((-y-x) .<= 0.8).*((-y-x) .>= 0.4).*(-x .+ 0.1)  + 5*(-x+y .< 1).*(-x+y .>= 0.8).*(abs.(x) .<= 1).*(abs.(y) .<= 1) + ( 50*0.25 .- 50*sqrt.((x.-0.5).^2+(y.+0.6).^2)).*(sqrt.((x.-0.5).^2+(y.+0.6).^2) .<= 0.25);

Random.seed!(1)

noisevar = 0.05

Nbig = 100
dimbig = Nbig
N = 64
dim = N
Nmeas = 50
dimmeas = Nmeas

xsbig = -1+1/dimbig:2/dimbig:1-1/(dimbig)
ysbig = -1+1/dimbig:2/dimbig:1-1/(dimbig)

xs = -1+1/dim:2/dim:1-1/(dim)
ys = -1+1/dim:2/dim:1-1/(dim)

xibig = linspace(-1,1,dimbig)
Ybig,Xbig = meshgrid(xibig,xibig)

ximeas = linspace(-1,1,dimmeas)
Ymeas,Xmeas = meshgrid(ximeas,ximeas)
Y,X = meshgrid(xs,xs)


Zbig = tf(Xbig,Ybig)
@time Fbig = measurementmatrix2d(Xbig,Ybig,Xmeas,Ymeas,kernel,constant=cw); # Theory matrix.
@time F = measurementmatrix2d(X,Y,Xmeas,Ymeas,kernel,constant=cw); # Theory matrix. 
Fm = maximum(F)
F[F .< 0.0001*Fm] .= 0
F = sparse(F)

#imshow(Zbig)

zx = Fbig*Zbig[:]
mebig = reshape(zx,(dimmeas,dimmeas))
y = mebig[:] + randn(Nmeas*Nmeas,)*sqrt(noisevar)


# # px=hcat([[y,x] for y in ys, x in xs]...)'[:,2]
# # py=hcat([[y,x] for y in ys, x in xs]...)'[:,1]
# # M = [x*y for y = ys, x = xs]
# itp = interpolate(mebig,BSpline(Quadratic( Line(OnCell() ) ) ) )
# itp = scale(itp,ysbig,xsbig)
# itp = extrapolate(itp,Line())
# meas = itp(ys,xs)
# p(x) = tt(x,argi)[1]
# grtt = gradi(p,x0)

Dx,Dy,Db = regmatrices_second(N); Dx = sparse(Dx); Dy = sparse(Dy); Db = sparse(Db)
S = sparse(spdematrix(xs,0.001,0.1))
D = [Dx;Dy;Db]
argi  = (Dx = Dx, Dy = Dy, D=D, F = F,noisesigma = sqrt(noisevar), scale = 0.3, y = y, bscale = 1.0, Db = Db )
argis  = ( D=S, F = F,noisesigma = sqrt(noisevar), scale = 0.3, y = y, bscale = 1.0 )

cacheiso =(xprop=zeros(length(N*N)),Fxprop=zeros(Nmeas^2),Dxprop=zeros(size(Dx)[1]),Dyprop=zeros(size(Dy)[1]),Dbprop=zeros(size(Db)[1]), residual=similar(y),gradiprop=zeros(N*N))
cached =(xprop=zeros(length(N*N)),Fxprop=zeros(Nmeas^2),Dprop=zeros(size(D)[1]), residual=similar(y),gradiprop=zeros(N*N))
caches =(xprop=zeros(length(N*N)),Fxprop=zeros(Nmeas^2),Dprop=zeros(size(S)[1]), residual=similar(y),gradiprop=zeros(N*N))


# h(x) = logpisodiff(x,argi,cacheiso)
# x0 = randn(N*)
# gr = gradi(h,x0)
# gr2 = logpisodiffgradi(x0,argi,cacheiso)[2]

# target1iso(x) = -logpisodiff(x,argi,cacheiso)# -logpdiff(w,diff1arg,cached)## 
# target1isograd(x) = -logpisodiffgradi(x,argi,cacheiso;both=false)[2]#  -logpdiffgradi(w,diff1arg,cached;both=false)[2]#
# res = Optim.optimize(target1iso, target1isograd, 0*randn(N*N,),Optim.LBFGS(), Optim.Options(allow_f_increases=true,show_trace=true,iterations=700); inplace = false)
# MAP_diff1iso = res.minimizer
# imshow(reshape(MAP_diff1iso,(N,N))); colorbar(); clim(-0.5,12)

# target1(x) = -logpdiff(x,argi,cached)# -logpdiff(w,diff1arg,cached)## 
# target1grad(x) = -logpdiffgradi(x,argi,cached;both=false)[2]#  -logpdiffgradi(w,diff1arg,cached;both=false)[2]#
# res = Optim.optimize(target1, target1grad, 0*randn(N*N,),Optim.LBFGS(), Optim.Options(allow_f_increases=true,show_trace=true,iterations=700); inplace = false)
# MAP_diff1 = res.minimizer
# imshow(reshape(MAP_diff1,(N,N))); colorbar(); clim(-0.5,12)

targets(x) = -logpdiff(x,argis,caches)# -logpdiff(w,diff1arg,cached)## 
targetsgrad(x) = -logpdiffgradi(x,argis,caches;both=false)[2]#  -logpdiffgradi(w,diff1arg,cached;both=false)[2]#
res = Optim.optimize(targets, targetsgrad, 0*randn(N*N,),Optim.LBFGS(), Optim.Options(allow_f_increases=true,show_trace=true,iterations=700); inplace = false)
MAP_s = res.minimizer
imshow(reshape(MAP_s,(N,N))); colorbar(); clim(-0.5,12)


# imshow(reshape(MAP_diff1iso,(N,N)))
# colorbar()
# clim(-0.5,8)

#imshow(Zbig); colorbar()