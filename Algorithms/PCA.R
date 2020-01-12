setwd(dirname(rstudioapi::getSourceEditorContext()$path))

library("FactoMineR")
library("corrplot")
library("factoextra")
library("fields")



visa_cleaned <- read.csv(file = "../Datasets/VisaPremier_cleaned.txt", row.names=1)
dim(visa_cleaned)
data_x <- visa_cleaned[, -dim(visa_cleaned)[2]]
data_y <- visa_cleaned[, dim(visa_cleaned)[2]]



dim(data_x)
length(data_y)

quali_names <- c("departem", "ptvente", "sexe", "sitfamil", "csp", "sexer", "codeqlt")
quali_sup_indexes <- c(1:length(names(visa_cleaned)))[names(visa_cleaned) %in% quali_names]
#ind_sup_indexes <- c(1:length(rownames(visa_cleaned)))[rownames(visa_cleaned) == "442153"]
label_col <- as.factor(data_y)

important_features <- c("moycred3", "avtscpte", "anciente", "engagemt", "agemvt", "nbcb", "mtfactur")


#####################
# PCA
#####################
resPCA <- PCA(data_x, quali.sup=quali_sup_indexes)
resPCA <- PCA(data_x)

fviz_pca_ind(resPCA, col.ind = label_col, label = "none", addEllipses = TRUE, xlim=c(-7.5,20))
fviz_pca_var(resPCA, repel = TRUE)

fviz_pca_var(resPCA, repel = TRUE, col.var = "contrib", gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"))

#plot.PCA(resPCA, choix = "var")
#plot.PCA(resPCA, choix = "ind", col.ind = label_col, label = "none")



#####################
# AFDM
#####################
quali_todo <- c("departem", "ptvente", "sitfamil", "csp", "sexer", "codeqlt")
famd_data_x <- data_x

for(quali in quali_todo){
  famd_data_x[quali] <- lapply(famd_data_x[quali], FUN=function(x){ paste(quali, x, sep="_") })  
}

famd_data_x

#famd_data_x[,! names(famd_data_x) %in% quali_todo] <- scale(famd_data_x[,! names(famd_data_x) %in% quali_todo])

resFAMD <- FAMD(famd_data_x)
resFAMD$quali.var

fviz_famd_ind(resFAMD, label = "none", col.ind = label_col, addEllipses = T)

fviz_famd_var(resFAMD, repel = TRUE, col.var = "black")

fviz_famd_var(resFAMD, "quanti.var", repel = TRUE, col.var = "black")
fviz_famd_var(resFAMD, "quali.var", repel = TRUE, col.var = "black")




fviz_famd_var(resFAMD, "quanti.var", col.var = "contrib", 
              gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
              repel = TRUE, labelsize = 6)

fviz_famd_var(resFAMD, "quali.var", col.var = "contrib", 
              gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
              repel = TRUE, labelsize = 5)
