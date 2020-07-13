library(ggridges)
library(ggplot2)
library(viridis)
library(hrbrthemes)
library(reshape2)
library(hash)
library(dplyr)

#### load the data
columns = c('ICU-stay', 'Hospital-Admission-to-death', 'onset-to-death', 'onset-to-diagnosis', 'onset-to-hospital-admission',
            'onset-to-hospital-discharge', 'onset-to-ICU-admission', 'onset-to-diagnosis-pcr')

map <- c('AC'='North', 'AM'='North', 'AP'='North', 'PA'='North', 'RO'='North', 'RR'='North', 'TO'='North',
         'AL'='Northeast', 'BA'='Northeast', 'CE'='Northeast', 'MA'='Northeast',
         'PB'='Northeast', 'PI'='Northeast', 'PE'='Northeast', 'SE'='Northeast', 'RN'='Northeast',
         'DF'='Central-West', 'GO'='Central-West', 'MS'='Central-West', 'Central-West',
         'ES'='Southeast', 'MG'='Southeast', 'RJ'='Southeast', 'SP'='Southeast',
         'PR'='South', 'RS'='South', 'SC'='South')


path <- '../results/MeansPosteriors/'

ICUstay <- read.csv(file = paste(path, columns[1], '.csv', sep = ''))
AdminDeath <- read.csv(file = paste(path, columns[2], '.csv', sep = ''))
oDeath <- read.csv(file = paste(path, columns[3], '.csv', sep = ''))
oDiagnosis <- read.csv(file = paste(path, columns[4], '.csv', sep = ''))
oAdmin <- read.csv(file = paste(path, columns[5], '.csv', sep = ''))
oDischarge <- read.csv(file = paste(path, columns[6], '.csv', sep = ''))
oICU <- read.csv(file = paste(path, columns[7], '.csv', sep = ''))
oDiagnosisPCR <- read.csv(file = paste(path, columns[8], '.csv', sep = ''))


#### prepare the dataframes

convert_df <- function(df_in){
  df <- data.frame(df_in)
  df <- melt(df)
  df <- df %>%
    mutate(geo = map[variable])
  df$geo <- as.factor(df$geo)
  a <- aggregate(x = df$value,
                 by = list(df$variable),
                 FUN = mean)
  aord <- a[order(a$x),]
  target <- as.vector(aord$Group.1)
  df <- df[order(match(df$variable, target)),]
  df$variable <- factor(df$variable, levels = target)
  #rownames(df) <- 1:nrow(df)
  rownames(df) <- NULL
  #print(head(df))
  return(df)
}

ICUstay2 <- convert_df(ICUstay)
AdminDeath2 <- convert_df(AdminDeath)
oDeath2 <- convert_df(oDeath)
oDiagnosis2 <- convert_df(oDiagnosis)
oAdmin2 <- convert_df(oAdmin)
oDischarge2 <- convert_df(oDischarge)
oICU2 <- convert_df(oICU)
oDiagnosisPCR2 <- convert_df(oDiagnosisPCR)


#### Plot


plot <- function(df, title) {
  if (title == 'Hospital-Admission-to-death'){title = 'Admission-death'}
  if (title == 'onset-to-diagnosis'){title = 'Onset-non-PCR-diagnosis'}
  if (title == 'onset-to-ICU-admission'){title = 'Onset-ICU-admission'}
  if (title == 'onset-to-hospital-admission'){title = 'Onset-admission'}
  if (title == 'onset-to-diagnosis-pcr'){title = 'Onset-PCR-diagnosis'}
  if (title == 'onset-to-death'){title = 'Onset-death'}
  if (title == 'onset-to-hospital-discharge'){title = 'Onset-discharge'}
  
  
  xmin <- quantile(df$value, probs = c(0.001)) #min(df$value)
  xmax <- quantile(df$value, probs = c(0.999)) #max(df$value)
  # could also be fill = stat(x)
  ggplot(df, aes(x = value, y = variable, fill = geo)) +
    geom_density_ridges_gradient(scale = 4, rel_min_height = 0.01) +
    labs(title=title, y = 'state', x = 'mean time (days)') +
    xlim(xmin, xmax) +
    theme(
      plot.title = element_text(color="black", size=10),
      legend.position="none",
      panel.spacing = unit(0.2, "lines"),
      strip.text.x = element_text(size = 4)
    )
}

g1 <- plot(ICUstay2, columns[1])
g2 <- plot(AdminDeath2, columns[2])
g3 <- plot(oDeath2, columns[3])
g4 <- plot(oDiagnosis2, columns[4])
g5 <- plot(oAdmin2, columns[5])
g6 <- plot(oDischarge2, columns[6])
g7 <- plot(oICU2, columns[7])
g8 <- plot(oDiagnosisPCR2, columns[8])


require(gridExtra)
### save the plots
grid.arrange(g5, g6, g8, g4, ncol=4, nrow=1)
grid.arrange(g3, g2, g7, g1, ncol=4, nrow=1)
