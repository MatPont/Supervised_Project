setwd(dirname(rstudioapi::getSourceEditorContext()$path))

library(caret)



#################################################
# Functions
#################################################
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}



#################################################
# Synthetic Data Sets
#################################################
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





#################################################
# Visa Premier
#################################################

#####################
# Pre-processing
#####################
# Load dataset
visa <- read.csv("../Datasets/VisaPremier.txt", sep="\t", row.names=1)
dim(visa)
# Arange dataset to have label as the last column
temp_names <- c(names(visa)[names(visa) != "cartevpr"], "cartevpr")
visa <- cbind(visa[,names(visa) != "cartevpr"], visa[, names(visa) == "cartevpr"])
names(visa) <- temp_names
dim(visa)


# Features selection
to_remove <- c()
for(i in 1:dim(visa)[2]){
  if(length(visa[,i]) == length(unique(visa[,i])) | length(unique(visa[,i])) == 1){
    to_remove <- c(to_remove, names(visa)[i])
  }
  else{
    print(names(visa)[i]) 
    if(length(table(visa[,i])) == 2)
      print(table(visa[,i]))    
  }
}

to_remove <- c(to_remove, "sexe", "cartevp", "nbbon", "mtbon") # These variables appears twice (with character values and binary values)

to_remove
visa_cleaned <- visa[, ! (names(visa) %in% to_remove)]
visa_cleaned <- visa_cleaned[-2, ]
dim(visa_cleaned)


# Manage NA values
sum(visa_cleaned == ".")
names(visa_cleaned)[apply(visa_cleaned, MARGIN=2, FUN=function(x){ "." %in% x })]

visa_cleaned[visa_cleaned[,"departem"] == ".", "departem"] <- getmode(visa_cleaned[,'departem'])
visa_cleaned[visa_cleaned[,"codeqlt"] == ".", "codeqlt"] <- getmode(visa_cleaned[,'codeqlt'])
visa_cleaned[visa_cleaned[,"agemvt"] == ".", "agemvt"] <- median(as.numeric(visa_cleaned[,'agemvt']))
visa_cleaned[visa_cleaned[,"nbpaiecb"] == ".", "nbpaiecb"] <- median(as.numeric(visa_cleaned[,'nbpaiecb']))

sum(visa_cleaned == ".")

write.csv(file = "../Datasets/VisaPremier_cleaned.txt", visa_cleaned)



#####################
# Preliminary Study
#####################

#variables <- c("nbop", "mteparmo", "codeqlt", "nbcb", "ndjdebit")

for(variable in names(data_x)){
  print(variable)
  boxplot(as.numeric(data_x[data_y == 0,variable]), as.numeric(data_x[data_y == 1,variable]))
  readline()
}


#################################################
# Credit Card Fraud
#################################################
