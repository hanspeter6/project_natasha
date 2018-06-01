# libraries:
library(dplyr)
library(FactoMineR)
library(factoextra)
library(rpart)
library(rpart.plot)
library(caret)
# library(logistf)
library(brglm)
library(corrplot)
library(mice)

# read .csv file:
geo_data <- read.table("trial.csv",
                       header = TRUE, sep = ";")

#change site, type, burn to factor variables
geo_data$site <- factor(geo_data$site)
geo_data$type <- factor(geo_data$type)
geo_data$burn <- ifelse(geo_data$burn == 1, 0, 1) # 0 = burn, 1 = no burn, swapped around
geo_data$burn <- factor(geo_data$burn, labels = c("NoBurn", "Burn"))

summary(geo_data)

# numeric dataset
geo_data_nums <- data.frame(scale(geo_data[,c('elevation',
                                              'slope',
                                              'hsv', # 3 NA's
                                              'moisture',
                                              'ph',
                                              'ig_loss',
                                              'sand',
                                              # 'silt', # due to singularity
                                              'clay')]))

summary(geo_data_nums) # 3 NAs

# impute missing values: multivariate imputations by chained equations (mice)
# which rows (hp row 35; hsv rows 35,64 & 79 )
which(is.na(rowSums(geo_data_nums)))
imputed <- mice(geo_data)
imputed$imp$hsv
imputed$imp$hp
pool(imputed)

# consider correlations
cor_mat <- cor(geo_data_nums, use = 'pairwise.complete.obs')
jpeg("corrplot1_geo.jpeg")
corrplot(cor_mat, method = 'pie', tl.col = "black")
dev.off()

# to explore dimensionality of numerical variables for now only complete cases
pca_geo <- princomp(geo_data_nums[complete.cases(geo_data_nums),])
summary(pca_geo)

jpeg("pca_screeplot.jpeg")
screeplot(pca_geo, type = "lines", main = "Scree Plot of Principal Components", npcs = 8)
dev.off()

loadings(pca_geo)

# using factomineR
pca_geo_factoMineR <- PCA(geo_data_nums[complete.cases(geo_data_nums),], ncp = 4, graph = FALSE)

jpeg("contributions_pca_1n2.jpeg")
fviz_pca_var(pca_geo_factoMineR,
             axes = c(1,2),
             col.var="contrib",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             title = "Dimensions 1 & 2",
             repel = TRUE # Avoid text overlapping
)
dev.off()

jpeg("contributions_pca_3n4.jpeg")
fviz_pca_var(pca_geo_factoMineR,
             axes = c(3,4),
             col.var="contrib",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             title = "Dimensions 3 & 4",
             repel = TRUE # Avoid text overlapping
)
dev.off()

# consider classification tree explantory:
tree_geo <- rpart(burn ~ 
                          area +
                          geology +
                          aspect +
                          elevation +
                          slope +
                          hsv +
                          hp +
                          moisture +
                          ph +
                          ig_loss +
                          colour +
                          sand +
                          silt +
                          clay
                          ,
                  data = geo_data)
par(xpd = NA)
plot(tree_geo)
text(tree_geo, pos = 2, cex = 0.8)
jpeg("partition_tree.jpeg")
rpart.plot(tree_geo, type = 4, extra = 1, cex = 0.8, main = "Burn Partition Tree")
dev.off()

# consider prediction power of single tree:
pred_class <- predict(tree_geo, type = "class")
confusionMatrix(data = pred_class, reference = geo_data$burn)

# Logistic regression (first round shows up perfect separation...)
lr_geo <- glm(burn ~ area +
                      geology +
                      aspect +
                      elevation +
                      slope +
                      hsv +
                      hp +
                      moisture +
                      ph +
                      ig_loss +
                      colour +
                      sand +
                      silt,
              data = geo_data,
              family = "binomial")

# perfect separation problems, so cant use ML methods
# using different "bias reduction" as per Firth (1993).
# the estimates may not be unbiased, but at least they are finite. 
# His method effectively removes a term in the estimation of coefficients.
# still want to reconsider stuff...with adjusted data.

alt_mod <- brglm(burn ~ 
                         # area +
                         #  geology +
                         # aspect +
                         # elevation +
                         # slope +
                         hsv +
                         hp +
                         # moisture +
                         ph +
                         ig_loss +
                         # colour +
                         sand,#´+
                 # silt,
                 data = geo_data,
                 family = "binomial")

summary(alt_mod)

#consider stepwise approaches to consider "best" models: 
library(glmnet)

# getting model.matrix
mat_mod <- model.matrix(burn ~ area +
                                geology +
                                aspect +
                                elevation +
                                slope +
                                hsv +
                                hp +
                                moisture +
                                ph +
                                ig_loss +
                                colour +
                                sand +
                                silt,
                        data = geo_data)

# find out NA rows
which(is.na(rowSums(geo_data_nums)))
net_mod <- glmnet(x = mat_mod[,-1], y = geo_data$burn[-c(35,64,79)],
                   family = "binomial")

plot(net_mod, label = TRUE)
summary(net_mod)
# 


# # consider causes of perfect separation:
# library(safeBinaryRegression)
# tester <- glm(burn ~ area +
#                       geology +
#                       aspect +
#                       elevation +
#                       slope +
#                       hsv +
#                       hp +
#                       moisture +
#                       ph +
#                       ig_loss +
#                       colour +
#                       sand +
#                       silt,
#               data = geo_data,
#               family = "binomial")
# 
# #The following terms are causing separation among the sample points:
# # (Intercept), areaCH, areaCP, areaGC, areaGiantC, areaIN, areaKB, areaMC, areaRN,
# #  geologyDolerite, geologyElliot, geologyMolteno, geologyTarkastad, aspectNorth,
# #   aspectNortheast, aspectSouth, aspectSouthwest, elevation, slope, hsv, hp, moisture,
# #    ph, ig_loss, colourDB, colourDGB, colourDYB, colourVDG, colourVDGB, sand, silt
# 
# ### EXAMPLES to explain what this means
# # ## A set of 4 completely separated sample points ##
# # x <- c(-2, -1, 1, 2)
# # y <- c(0, 0, 1, 1)
# # glm(y ~ x, family = binomial)
# # 
# # plot(x,y)
# # ## A set of 4 quasicompletely separated sample points ##
# # x <- c(-2, 0, 0, 2)
# # y <- c(0, 0, 1, 1)
# # glm(y ~ x, family = binomial)
# # plot(x,y)
# 
# detach("package:safeBinaryRegression", unload = TRUE)
