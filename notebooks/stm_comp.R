install.packages('lda')
install.packages('slam')
install.packages('stm')
install.packages('igraph')
install.packages('stmCorrViz')
library(lda)
library(slam)
library(stm)
library(igraph)     # Package for network analysis and visualisation
library(stmCorrViz) # Package for hierarchical correlation view of STMs

data <- read.csv("/work/socialmedia/metoo/merged_company_metoo.csv") 
#data <-read.csv("components/artifacts/postprocess_data.csv", sep = '\t')
head(data)
tmp <- data[c('Full.Text','src_flag')]
#load("VignetteObjects.RData") 

processed <- textProcessor(tmp$Full.Text, metadata=tmp)
out <- prepDocuments(processed$documents, processed$vocab, processed$meta)

docs <- out$documents
vocab <- out$vocab
meta <- out$meta

plotRemoved(processed$documents, lower.thresh=seq(1,200, by=100))

# K -> num of topics
oliblogPrevFit <- stm(out$documents, out$vocab, K=10, prevalence=~src_flag, 
                      max.em.its=75, data=out$meta, init.type="Spectral", 
                      seed=8458159)

png("stm_company.png")
plot(oliblogPrevFit, type="summary", xlim=c(0,.4))
dev.off() 
