import Pkg;
#Pkg.add("Clustering")
#Pkg.add("Plots")
Pkg.add("GaussianMixtures")

using LinearAlgebra
using Clustering
using Plots
using GaussianMixtures

##TODO: define the algorithm for a sigma
function assign_k_clusters(k,sigma,points) #points are given as matrix P, each point in row

    #1. Build the affinity matrix
    AffMat=zeros(length(points),length(points))
    for i in 1:length(points)
        for j in 1:length(points)
            if (i!=j)
                AffMat[i,j]=exp(LinearAlgebra.norm(points[i]-points[j],2))/(2*sigma^2)
            end
        end
    end

    scatter(AffMat[:,1], AffMat[:,2],
        color=:lightrainbow, legend=false)

    savefig("~/desktop/code_vignettes/10_2_2020/clustering_trials/Aff_dist.png")

    #2. Define D
    RowSums=zeros(length(points))
    for i in 1:length(points)
        RowSums[i]=sum(AffMat[i,:])
    end
    DMat=LinearAlgebra.Diagonal(RowSums)

    #3. Define Laplacian
    LMat=(DMat^(-1/2))*AffMat*(DMat^(-1/2))

    scatter(LMat[:,1], LMat[:,2],
        color=:lightrainbow, legend=false)

    savefig("~/desktop/code_vignettes/10_2_2020/clustering_trials/L_dist.png")

    #4. Form X by stacking k largest eigenvectors
    X=eigen(LMat).vectors[:,(length(points)-k+1):length(points)] #; permute::Bool=true, scale::Bool=true, sortby

    #println("Eigens")
    

    #5. Form Y by renormalizing X so that each row has unit length
    Y=zeros(length(points),k)

    for j in 1:k
        Y[:,j] = Y[:,j]+ X[:,j] ./ sqrt(sum(X[:,j] .^ 2))
    end

    scatter(Y[:,1], Y[:,2],
        color=:lightrainbow, legend=false)

    savefig("~/desktop/code_vignettes/10_2_2020/clustering_trials/Y_dist.png")
    #6. Perform k-means on rows (transpose as kmeans uses columns as points)
    YR=kmeans(transpose(X),k)
    YAssgn=assignments(YR)

    scatter(X[:,1], X[:,2],marker_z=YAssgn,
        color=:lightrainbow, legend=false)

    savefig("~/desktop/code_vignettes/10_2_2020/clustering_trials/X_dist.png")

    #7. Return assignments for points
    #println(YAssgn)

    #8. 
    return YAssgn

end

##TODO: build cheeger constant

##TODO: build cheeger approximation: Assign every node to a dimension. Then, assign every node to a point in the space. Do a random
#graph walk from each node, and on the nth step add 1/n to the point corresponding to the beginning
#node, in the dimension of the point. Then, do a distance measure on the clusters generated.

##TODO: define sigma discovery algorithm (want to maximize the smallest Cheeger Constant)

##TODO: Find clustering algo to use that's better than knn 
##to end the spectral method. GMM sounds good
function reparam(x;dim=2)
    y=zeros(dim)
    y[1]=cos(2*pi*x[1])
    y[2]=sin(2*pi*x[1])
    return y
end
##TODO: build test
k=2
sigma=500
points=[(i%2)*reparam(rand(1,1)) + 3*((i+1)%2)*(reparam(rand(1,1))) for i in 1:500]
P=zeros(2,500)
for p in 1:500
    P[:,p]=points[p]
end
assignment=assign_k_clusters(k,sigma,points)

scatter(P[1,:], P[2,:], marker_z=assignment,
        color=:lightrainbow, legend=false)

savefig("~/desktop/code_vignettes/10_2_2020/clustering_trials/spectral_trial.png")

##do straightforward knn as comparison
R=kmeans(P,k)
scatter(P[1,:], P[2,:], marker_z=assignments(R),
        color=:lightrainbow, legend=false)

savefig("~/desktop/code_vignettes/10_2_2020/clustering_trials/vanilla_knn.png")
