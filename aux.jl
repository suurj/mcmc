
#module extras
    using LinearAlgebra
    using Random
    using ProgressBars

   @inline function meshgrid(x,y) 
        grid_a = [i for i in x, j in y]
        grid_b = [j for i in x, j in y]
    
        return grid_a,grid_b
    end

    function orthonormalbase(A)
        r = rank(A)
        U, S, V = svd(A)
        return U[:,1:r]
      end
      

@inbounds  function measurementmatrix2d(X,Y,kernel;constant=10.0)
    dy = Y[2]-Y[1];
    dx = X[2]-X[1];
    d = Float64(dy*dx)
    Ny = length(Y); Nx = length(X)
    F = zeros(Ny*Nx,Ny*Nx)

     for i = 1:Ny*Nx
        for j = 1:i
            xi = div(i-1,Ny)+1; 
            yi = rem(i,Ny)+1
            xj = div(j-1,Ny)+1; 
            yj = rem(j,Ny)+1
            F[i,j] = d*kernel(X[xi],X[xj],Y[yi],Y[yj],constant=constant);
            F[j,i] = F[i,j]
        end
    end

    return F

end

function  s2dmatrix(x0,a,b)
    N = length(x0);
    dx = x0[2]- x0[1]
   
   M = zeros(N,N);
   for i=2:N-1
       M[i,i-1] = -a/dx^2;
       M[i,i+1] = -a/dx^2;
       M[i,i] = 2*a/dx^2;

   end
   M[1,:] = [2*a/dx^2 -a/dx^2 zeros(1,N-3) -a/dx^2];
   M[N,:] = [-a/dx^2 zeros(1,N-3) -a/dx^2 2*a/dx^2];
   
   M = kron(M,I(N)) + kron(I(N),M)
   M = M + I(size(M)[2])*b;

   return M
end

    function linspace(start,stop,length)
        return Vector(range(start,stop=stop,length=length))
    end

    @inline function alphastable(alpha::Float64,scale::Float64)::Float64
        W = randexp()
        U = rand()*pi - pi/2
        return  scale*sin(alpha*U)/cos(U)^(1/alpha)*(cos(U-alpha*U)/W)^((1-alpha)/alpha)
    end

    @inline function dotitself(x::Array{Float64,1})  
        return dot(x,x)
    end

    @inline function dotitself(x::Array{Complex{Float64},1})
        return transpose(x)*x
    end

    @inline function logsigmoid(x::Float64)
        if x >= 0
           z = exp(-x)
           return 1 / (1 + z)
       else
           z = exp(x)
           return z / (1 + z)
       end
    end

    @inline function logsigmoid(x::ComplexF64)
        if real(x) >= 0
           z = exp(-x)
           return 1 / (1 + z)
       else
           z = exp(x)
           return z / (1 + z)
       end
    end

    @inline function logit(x)
        return log(x/(1-x))
    end


    @inline function log1pexp(x::Float64)
        if (x<=-37.0)
            return exp(x)
        elseif (-37<x<=18)
            return log1p(exp(x))
        elseif (18<x<=33.3)
            return x + exp(-x)
        else
            return x
        end

    end

    @inline function log1pexp(x::ComplexF64)
        if (real(x)<=-37.0)
            return exp(x)
        elseif (-37<real(x)<=18)
            return log1p(exp(x))
        elseif (18<real(x)<=33.3)
            return x + exp(-x)
        else
            return x
        end

    end

    #export sampler
    mutable struct sampler
        Cho::Cholesky{Float64,Array{Float64,2}}
        acc::Int64
        cgr::Vector{Int64}
    end

    function divide(N,NT)
        even = mod(N,NT) == 0
        k = Int64(floor(N/NT))
        t = 0
        if(even)
            q = Vector{Vector{Int64}}(undef,k)
            t = k

        else
            q = Vector{Vector{Int64}}(undef,k+1)
            t = k +1
        end

        for i = 1:t
            q[i] = Vector(NT*(i-1)+1:min(NT*i,N))
        end

        return q


    end

    function comporam(;
        N = 1000,
        x0 = zeros(2,),
        cgr = nothing,
        logpdf = x -> (-0.5 * x' * x),
        adaptruns = 2,
        Nadapt = 1500,
        thinning = 1,
        C = nothing,
        atarg=0.3,
    )
        x = copy(x0)
        dim = size(x)[1]

        if (isnothing(C))
            C = 1.5
        end

        if (isnothing(cgr))
            cgr = Vector([
                Vector(1:trunc(Int, dim / 2)),
                Vector(trunc(Int, dim / 2)+1:dim),
            ])
        end

        clistsize = size(cgr)[1]
        slist = Vector{sampler}(undef, clistsize)

        for i = 1:clistsize
            ns = size(cgr[i])[1]
            slist[i] = sampler(
                cholesky(Hermitian(C * Array(Diagonal(ones(ns, ns))))),
                0,
                cgr[i],
            )
        end
        cgr = nothing

        chain = zeros(dim, Int(floor(N / thinning)))
        density = logpdf(x)
        #auxnew = copy(aux)
        #auxchain = zeros(length(aux), Int(floor(N / thinning)))
        gamma = 0.8
        xn = copy(x)

        for ar = 1:adaptruns
            totalacc = 0.0
            for k = 1:clistsize
                slist[k].acc = 0
            end
            for i in ProgressBar(1:Nadapt)
                for k = 1:clistsize
                    compos = slist[k].cgr
                    nc = size(compos)[1]
                    R = slist[k].Cho.L
                    eta = i^(-gamma)
                    u = randn(nc)
                    res = R*u
                    partialsum!(xn, x, res, compos)
                    densitynew = logpdf(xn)
                    ratio = densitynew - density
                    if (log(rand()) <= ratio)
                        x[compos] += res
                        density = densitynew
                        totalacc = totalacc + 1
                        slist[k].acc = slist[k].acc + 1
                    end

                    u = u / norm(u)
                    z =
                        sqrt(eta * abs(slist[k].acc / i - atarg)) *
                        slist[k].Cho.L *
                        u
                    if (slist[k].acc / i >= atarg)
                        lowrankupdate!(slist[k].Cho, z)
                    else
                        lowrankdowndate!(slist[k].Cho, z)
                    end

                end
            end
            println("Adaptation stage no. ",ar," acceptance ratio: ",totalacc / (Nadapt*clistsize))
        end

        totalacc = 0.0

        for i in ProgressBar(1:N)
            for k = 1:clistsize
                compos = slist[k].cgr
                nc = size(compos)[1]
                R = slist[k].Cho.L
                u = randn(nc)
                res = R*u
                partialsum!(xn, x, res, compos)
                densitynew = logpdf(xn)
                ratio = densitynew - density
                if (log(rand()) <= ratio)
                    x[compos] += res
                    #aux = auxnew
                    density = densitynew
                    totalacc = totalacc + 1
                end
            end
            if (i % thinning == 0)
                chain[:, Int(i / thinning)] = x
                #auxchain[:, Int(i / thinning)] = aux
            end

        end

        println("Sampling stage acceptance ratio: ", totalacc / (N*clistsize))
        return chain
    end


    function logsumexp(w)
        offset = maximum(real(w))
        we = exp.(w .- offset)
        s = sum(we)
        return log(s) + offset
    end

    function logsumexp!(c,w)
        offset = maximum(real(w))
        c .= w .- offset
        c .= exp.(c)
        s = sum(c)
        return log(s) + offset
    end

    function softmax!(out,w)
        out .= w .- maximum(w)
        out .= exp.(out)
        denominator = sum(out)
        out .= out/denominator
        return  nothing #out./denominator
    end

    function softmax(w)
        out = w .- maximum(w)
        out = exp.(out)
        denominator = sum(out)
        return  out./denominator
    end


    function gradic(f,x0;eps=1e-10)
        xp = ComplexF64.((x0))
        f0 = f(x0)
        Nc = length(f0)
        Nv = length(x0)
        if(Nc > 1)
            J = zeros(Nc,Nv)
        else
            J = zeros(Nv,)
        end
        for i = 1:Nv
            orig = xp[i]
            xp[i] += im*eps
            fp = f(xp)
            if(Nc > 1)
                J[:,i] .= imag.(fp)./eps
            else
                J[i] = imag(fp)/eps
            end
            xp[i] = orig
        end

        return J
    end

    function gradi(f,x0;eps=1e-8)
        xp = copy(x0)
        f0 = f(x0)
        Nc = length(f0)
        Nv = length(x0)
        if(Nc > 1)
            J = zeros(Nc,Nv)
        else
            J = zeros(Nv,)
        end
        for i = 1:Nv
            orig = xp[i]
            xp[i] += eps
            fp = f(xp)
            if(Nc > 1)
                J[:,i] .= (fp-f0)./eps
            else
                J[i] = (fp-f0)/eps
            end
            xp[i] = orig
        end

        return J
    end

    function partialsum!(z, s0, sp, spc)
        N = size(spc)[1]
         for i = 1:N
            r = spc[i]
            z[r] = s0[r] + sp[i]
        end
        return nothing
    end

    function ravel(q)
        return dropdims(q, dims = tuple(findall(size(q) .== 1)...))
    end

    function difference1(X)
        N = length(X);
        M = zeros(N,N);

        for i  = 2:N
           M[i,i] = -1;
           M[i,i-1] = 1;
        end
        M[1,1] = 1;
        return M
    end

    function  difference2(X)
        N = length(X);
        M = zeros(N,N);

        for i  = 3:N
           M[i,i] = -1;
           M[i,i-1] = 2;
           M[i,i-2] = -1;
        end
        M[1,1] = 1; M[2,1] = 1; M[2,2] = -1;
        return M

    end

    function measurementmatrix(X,MX,kernel)
        s = X[2]-X[1];
        F = zeros(size(MX)[1],size(X)[1])
        for i = 1:size(MX)[1]
            F[i,:] = s.*kernel.(X,MX[i]);
        end
        return F
    end

    function smatrix(X0,a,b)
        N = length(X0);
        dx = X0[2]-X0[1];

        M = zeros(N,N);
        for i=2:N-1
            M[i,i-1] = -a/dx^2;
            M[i,i+1] = -a/dx^2;
            M[i,i] = 2*a/dx^2 +b;

        end
        #M[1,:] = [b+2*a/dx^2 -a/dx^2 zeros(1,N-3) -a/dx^2];
        #M[N,:] = [-a/dx^2 zeros(1,N-3) -a/dx^2 b+2*a/dx^2];

        M[1,1] = 1;
        #M(N,1) = 1/dx;
        #M(N,2) = -1/dx;

        M[N,N] = 1
        return M

    end


#end
