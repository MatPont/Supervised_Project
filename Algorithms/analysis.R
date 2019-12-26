setwd(dirname(rstudioapi::getSourceEditorContext()$path))

flame = read.csv("../Datasets/flame.txt", header=F, sep="\t")
dim(flame)

spiral = read.csv("../Datasets/spiral.txt", header=F, sep="\t")
dim(spiral)

agg = read.csv("../Datasets/Aggregation.txt", header=F, sep="\t")
dim(agg)


layout(matrix(1:3, nrow=1))
plot(flame[,1:2], col=flame[,3], pch=20, main="Flame", xlab="", ylab="")
plot(spiral[,1:2], col=spiral[,3], pch=20, main="Spiral", xlab="", ylab="")
plot(agg[,1:2], col=agg[,3], pch=20, main="Aggregation", xlab="", ylab="")

