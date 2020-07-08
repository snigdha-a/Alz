

if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

if (!require("SummarizedExperiment", character.only = TRUE)) {
      install.packages("SummarizedExperiment")
      library("SummarizedExperiment", character.only = TRUE)
    }
# BiocManager::install("SummarizedExperiment")
obj <- readRDS("scATAC_Heme_All_SummarizedExperiment.final.rds")
show(obj)
colData(obj)
rowData(obj)
