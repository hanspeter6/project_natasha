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
geo_data <- read.table("trial2.csv",
                       header = TRUE, sep = ";")

#change site, type, burn to factor variables
geo_data$site <- factor(geo_data$site)
geo_data$type <- factor(geo_data$type)
geo_data$burn <- factor(geo_data$burn, labels = c("No Burn", "Burn"))

summary(geo_data)

# numeric dataset
geo_data_nums <- data.frame(scale(geo_data[,c('elevation',
                                              'slope',
                                              'hsv', # 3 NA's
                                              'hp',
                                              'moisture',
                                              'ph',
                                              'loss_ig',
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
lr_geo <- glm(burn ~ area2 +
                      geology2 +
                      aspect2 +
                      veg +
                      elevation +
                      slope +
                      hsv +
                      hp +
                      moisture +
                      ph +
                      loss_ig +
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

# focus on subset selection:

model_geo <- brglm(burn ~
                           aspect2 +
                           veg +
                           elevation +
                           hsv +
                           hp +
                           ph +
                           loss_ig +
                           sand,
                   data = geo_data,
                   family = "binomial")

summary(model_geo)

preds <- predict(model_geo, type = "response")
preds_num <- ifelse(preds > 0.5, 1, 0)
obs_num <- ifelse(geo_data$burn[-c(35,64,79)] == "Burn", 1, 0)
cor(preds_num, obs_num)^2

35,64 & 79

geo_data$burn
pchisq(deviance(model_geo), df.residual(model_geo), lower = FALSE)


summary(alt_mod_full)

library(bestglm)
inp <- geo_data %>%
        select( area2,
                geology2,
                aspect2,
                veg,
                elevation,
                slope,
                hsv ,
                hp,
                moisture,
                ph,
                loss_ig,
                colour,
                sand,
                clay,
                y = burn)
tester <- bestglm(inp, family = binomial, IC = "BIC", method = "exhaustive")

tester$BestModels
summary(tester$BestModel) #can use this to build model using brglm()

# try lassoo:
glmnet()


#Example 8. Logistic regression
data(SAheart)
bestglm(SAheart, IC="BIC", family=binomial)
#BIC agrees with backward stepwise approach
out<-glm(chd~., data=SAheart, family=binomial)
step(out, k=log(nrow(SAheart)))
#but BICq with q=0.25
bestglm(SAheart, IC="BICq", t=0.25, family=binomial)



# example with only ph as predictor:
# intercept = 8.283 : the log odds for burn with ph = 0.
# coefficient = -2.09 : the expected change in log odds for a one-unit increase in ph.
# the odds ratio can be calculated as exp(-2.09) = 0.1236871 :
#        so we would expect about an 82% decrease in the odds of being classified as burn for a one unit increase in ph

# want to confirm this:
# add column with predicted values:

geo_data$predicted <- predict(alt_mod) # log(odds) . Can use type = "response" for the predicted probabilities

#Examine the effect of a one-unit increase in ph by subtracting the corresponding log odds.
# For example, we will look at ph of 4.18 and 3.18 and calculate the difference in the estimated log odds.
#  Then exponentiate it to get the odds ratio.
s1<-geo_data$predicted[geo_data$ph==3.18][1]
s2<-geo_data$predicted[geo_data$ph==4.18][1]

logodd_diff<- s2 - s1
odd_ratio <- exp(logodd_diff)

xtabs(~ geology + area, data = geo_data)
# consider this with 

exp(confint(alt_mod))
exp(-3.99)
exp(0.3576)
predict(alt_mod, type = "response")

?predict
#consider stepwise approaches to consider "best" models: 
library(glmnet)

# getting model.matrix
mat_mod <- model.matrix(burn ~ area2 +
                                geology2 +
                                aspect2 +
                                veg +
                                elevation +
                                slope +
                                hsv +
                                hp +
                                moisture +
                                ph +
                                loss_ig +
                                colour +
                                sand +
                                clay,
                        data = geo_data)

# find out NA rows
which(is.na(rowSums(geo_data_nums)))
# Note alpha=1 for lasso only and can blend with ridge penalty down to
# alpha=0 ridge only.
net_mod <- glmnet(x = mat_mod[,-1], y = geo_data$burn[-c(35,64,79)],
                   family = "binomial", alpha = 1)

# Plot variable coefficients vs. shrinkage parameter lambda.
plot(net_mod, xvar="lambda", label = TRUE)

plot(net_mod, label = TRUE)
summary(net_mod)
net_mod 

coef(net_mod)[,8]

cv.glmmod <- cv.glmnet(x = mat_mod[,-1], y = geo_data$burn[-c(35,64,79)],
                       family = "binomial", alpha = 1)
plot(cv.glmmod)
(best.lambda <- cv.glmmod$lambda.min)

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


# workin on interpretations:

sample1 <- (46.9169601) + (-5.6199219)*(1) + (7.4460668)*(1) + (-0.0189997)*(1778.122) + (0.0478927)*(140) + (0.0010664)*(4200)+ (-7.5798635)*(3.88) + (0.2390717)*(9.759161) + ( 0.3659224)*(30.33467)

sample2 <- (46.9169601) + (-5.6199219)*(1) + (7.4460668)*(1) + (-0.0189997)*(1777.968) + (0.0478927)*(38) + (0.0010664)*(900) + (-7.5798635)*(3.89) + (0.2390717)*(23.636391) + ( 0.3659224)*(14.96546)

exp(sample1)
exp(sample2)

exp(sample1)/(1+exp(sample1))
exp(sample2)/(1+exp(sample2))

###MISSS
###
###
###
###
# explore imputed:
library(mice)
library(VIM)
imputed <- mice(geo_data, print = FALSE, m = 50)

# method = Predictive Mean Matching : check out: https://statisticalhorizons.com/predictive-mean-matching

imputed$imp$hsv
marginplot(geo_data[,c('hsv', 'hp')], col = mdc(1:2))

stripplot(imputed, hsv + hp ~ .imp, cex = 1.2, pch = 20)
fit <- with(imputed, brglm(burn ~
                                         aspect2 +
                                         veg +
                                         elevation +
                                         hsv +
                                         hp +
                                         ph +
                                         loss_ig +
                                         sand))
pooled <- pool(fit)
round(summary(pooled),4)
imputed$predictorMatrix

fitted(fit)