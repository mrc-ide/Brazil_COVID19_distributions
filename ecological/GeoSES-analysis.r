df = fread("~/Downloads/meansForRegression.csv")
df.var = fread("~/Downloads/varForRegression.csv")
df.scaled = scale(df[,2:23])
df.scaled = as.data.frame(df.scaled)
covars = read_excel("~/Downloads/journal.pone.0232074.s003.xls")
pop = fread("~/Downloads/caso_full.csv")
pop = pop[,list(pop=estimated_population_2019[1]),by=list(MUNIC_CODE7=city_ibge_code,state=state)]

covars2 = merge(pop,covars,by=c("MUNIC_CODE7"))


# GeoSES	the proposed socioeconomic index, varying from -1 to +1 (from the worst to the best conditions)
# GeoSESed	Education dimension (%)
# GeoSESpv	Poverty dimension (%)
# GeoSESdp	Deprivation dimension (%)
# GeoSESwl	Wealth dimension (%)
# GeoSESin	Income dimension (in Brazilian reais; 1 American dollar = 1.76 Brazilian reais in 2010)
# GeoSESsg	Dimension of segregation by race and income (Index of Concentration at the Extremes, varying from -1 to +1)

covars2 = covars2[, list(HDI=sum(HDI * pop) / sum(pop),
               `HDI education`=sum(HDI_educ * pop) / sum(pop),
               `HDI longevity`=sum(HDI_long * pop) / sum(pop),
               `HDI income`=sum(HDI_inc * pop) / sum(pop),
               GeoSES=sum(HDI_long * pop) / sum(pop),
               `GeoSES education` =sum(GeoSESed * pop) / sum(pop),
               `GeoSES poverty` =sum(GeoSESpv * pop) / sum(pop),
               `GeoSES deprivation` =sum(GeoSESdp * pop) / sum(pop),
               `GeoSES wealth` =sum(GeoSESwl * pop) / sum(pop),
               `GeoSES income` =sum(GeoSESin * pop) / sum(pop),
               `GeoSES segregation` =sum(GeoSESsg * pop) / sum(pop),
               n = length(pop)), by=state]

df3 = merge(covars2,df,by="state")
               
fit = stan_glm(`Hospital-Admission-to-death` ~ 
                   -1 + GeoSESed + GeoSESpv + GeoSESdp + GeoSESwl + GeoSESin
                 + GeoSESsg + mean_age + urban, data=df.scaled, prior=normal,
                 weights=1/df.var$`Hospital-Admission-to-death`)
plot(fit)

df3 = as.data.frame(df3)
df3 = as.data.frame(scale(df3[,c(2:16,28:35)]))
fit = stan_glm(`Hospital-Admission-to-death` ~ 
                 -1 + `GeoSES education` + `GeoSES poverty` + `GeoSES deprivation`
               + `GeoSES wealth` + `GeoSES income` + `GeoSES segregation` + mean_age + urban, data=df3, 
               weights=1/df.var$`Hospital-Admission-to-death`)
plot(fit)

fit = stan_glm(`onset-to-death` ~ 
                 -1 + `GeoSES poverty` + `GeoSES segregation` + `urban` + mean_age, data=df3, 
               weights=1/df.var$`onset-to-death`)
plot(fit)
fit = stan_glm(`onset-to-death` ~ 
                 -1 + `GeoSES deprivation` + `GeoSES segregation` , data=df3, 
               weights=1/df.var$`onset-to-death`)
plot(fit)

fit = lm(`Hospital-Admission-to-death` ~ 
                 -1 + `GeoSES education` + `GeoSES poverty` + `GeoSES deprivation`
               + `GeoSES wealth` + `GeoSES income` + `GeoSES segregation` + mean_age + urban, data=df3, 
               weights=1/df.var$`Hospital-Admission-to-death`)


summary(lm(`Hospital-Admission-to-death` ~ 
           -1 + `GeoSES deprivation` +`GeoSES wealth` + `GeoSES segregation`, data=df3, 
         weights=1/df.var$`Hospital-Admission-to-death`))

# p0 <- 4 # prior guess for the number of relevant variables
# n = nrow(df.scaled)
# p = 6
# tau0 <- p0/(p-p0) * 1/sqrt(n)
# hs_prior <- hs(df=1, global_df=1, global_scale=tau0)

# df2 = fread("~/Downloads/varForRegression.csv")
# fit1 = stan_lm(`Hospital-Admission-to-death` ~ 
#                 GeoSESed + GeoSESsg + mean_age, data=df.scaled, prior=R2(.5,what="mean"))
# plot(fit1)
# 
# fit2 = stan_lm(`Hospital-Admission-to-death` ~ 
#                  GeoSESed + GeoSESpv + GeoSESdp + GeoSESwl + GeoSESin
#                + GeoSESsg, data=df.scaled, prior=R2(.5,what="mean"))
# plot(fit2)
# df.var = fread("~/Downloads/varForRegression.csv")
# fit3 = stan_lm(`onset-to-diagnosis-pcr` ~ 
#                  GeoSESed + GeoSESpv + GeoSESdp + GeoSESwl + GeoSESin
#                + GeoSESsg, data=df.scaled, prior=R2(.5,what="mean"))
# plot(fit3)
# 
# p0 <- 4 # prior guess for the number of relevant variables
# n = nrow(df.scaled)
# p = 6
# tau0 <- p0/(p-p0) * 1/sqrt(n)
# hs_prior <- hs(df=1, global_df=1, global_scale=tau0)
# 
# fit4 = stan_glm(`Hospital-Admission-to-death` ~ 
#                    -1 + GeoSESed + GeoSESpv + GeoSESdp + GeoSESwl + GeoSESin
#                  + GeoSESsg, data=df.scaled, prior=hs_prior,
#                 prior_intercept=student_t(df = 7, location = 0, scale = 2.5),
#                prior_PD = TRUE,
#                weights=1/df.var$`Hospital-Admission-to-death`)
# plot(fit4)
# print(fit4$stanfit)
# fit5 = stan_glm(`Hospital-Admission-to-death` ~ 
#                   -1 + GeoSESed + GeoSESpv + GeoSESdp + GeoSESwl + GeoSESin
#                 + GeoSESsg, data=df.scaled, prior=normal(),
#                 prior_intercept=student_t(df = 7, location = 0, scale = 2.5),
#                 prior_PD = TRUE,
#                 weights=1/df.var$`Hospital-Admission-to-death`)

df4 = df3[,c(-12)]

cors = NULL
for(i in 15:22) {
  dd = cor(df4[, c(1:14)],df4[,i],use="complete")
  cors = cbind(cors,dd[,1])
}
cors = data.frame(cors)
names(cors) = names(df4)[15:22]
cors = round(cors,2)
View(cors)
library(stargazer)
stargazer(cors,digits=2,rownames=T,summary=F,out='outputs/GeoSES-correlations.tex')
