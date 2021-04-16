# if (!requireNamespace("BiocManager", quietly = TRUE))
#     install.packages("BiocManager")
# if (!requireNamespace("devtools", quietly = TRUE))
#     install.packages("devtools", repo="http://cran.rstudio.com/")


# install.packages("data.table", repos="https://Rdatatable.github.io/data.table")
# BiocManager::install("S4Vectors")
# install.packages("Rcpp", repos="https://rcppcore.github.io/drat")
# install.packages('plyr', repos='http://cran.us.r-project.org')
# install.packages("gtools", repos="https://cran.rstudio.com/")
# install.packages(c("ellipsis","withr","fansi"),repos="https://CRAN.R-project.org")
# install.packages("backports",repos="https://CRAN.R-project.org")
# install.packages("lattice",repos="https://CRAN.R-project.org")
# install.packages("Matrix",repos="https://CRAN.R-project.org")
# install.packages("nlme",repos="https://CRAN.R-project.org")
# BiocManager::install(c("ComplexHeatmap","rhdf5","motifmatchr","chromVAR","GenomeInfoDb","CNEr","DirichletMultinomial","seqLogo",
# "annotate","GO.db","KEGGREST","AnnotationDbi","DelayedArray","shiny","Rhdf5lib","TFBSTools"))
# install.packages(c("ggplot2","nabor","uwot","ggrepel", "farver","lifecycle","pillar","vctrs","RCurl","BH","caTools","DBI",
# "RSQLite","TFMPvalue","readr","reshape2","poweRlaw","R.utils","bit64","blob","bit","isoband","MASS","mgcv","scales","tibble",
# "pracma","R.methodsS3","hms","curl","mime","stringi","RcppEigen","htmltools","tidyr","hexbin","dplyr","crosstalk","purrr",
# "promises","generics","tidyselect","yaml","later","httpuv","GlobalOptions","shape","rjson","cluster","RcppArmadillo","plotly",
# "miniUI","RSpectra","RcppAnnoy","RcppProgress","circlize","GetoptLong","clue"),repos="https://CRAN.R-project.org")

# library(devtools)
# options(unzip = "/usr/bin/unzip")
# Sys.setenv(TAR = "/bin/tar")
# Sys.setenv(R_REMOTES_NO_ERRORS_FROM_WARNINGS="true")
# install_github("GreenleafLab/ArchR", ref="master", repos = c("https://CRAN.R-project.org",BiocManager::repositories()), verbose=TRUE)
suppressMessages(library(ArchR))
options(stringsAsFactors = F)
# ArchR::installExtraPackages()

#post installation
addArchRThreads(threads = 6)

addArchRGenome("hg19")
my_path <- "filtered_frag/"
# my_path <- "frag_out/"
# my_path <- "tmp_frag/"
# names <- c()
# inputFiles <- list.files(path = my_path)
ArrowFiles <- c()
barcodes <- scan('barcodes',what ='character', sep = "\n")
# for(i in 1:length(inputFiles))
# {
#   m = strsplit(inputFiles[i], '_')
#   # taking string between first and last _
#   name <- paste(head(tail(m[[1]],-1),-1),collapse="_")
#   # names <- append(names, name)
#   inputFiles[i] <- paste(my_path, inputFiles[i],collapse="", sep="")
#   ArrowFiles <- append(ArrowFiles, createArrowFiles(inputFiles = inputFiles[i],
#                                   sampleNames = name,
#                                   verbose=TRUE,
#                                   validBarcodes = barcodes,
#                                   minTSS = 2, #Don't set this too high because you can always increase later, changed from 0 to 3 by Irene
#                                   minFrags = 50, # minFrags changed from 0 to 500 by Irene
#                                   # nChunk increased to 20 by Irene, remove all but barcode after first colon from bamFile QNAME file
#                                   nChunk = 5, force = TRUE))
# }


# ArrowFiles <- createArrowFiles(inputFiles = inputFiles,
#                                 sampleNames = names,
#                                 verbose=TRUE,
#                                 validBarcodes = barcodes,
#                                 minTSS = 2, #Don't set this too high because you can always increase later, changed from 0 to 3 by Irene
#                                 minFrags = 50, # minFrags changed from 0 to 500 by Irene
#                                 # nChunk increased to 20 by Irene, remove all but barcode after first colon from bamFile QNAME file
#                                 nChunk = 5, force = TRUE)

ArrowFiles = list.files(path = '.',pattern="*.arrow")
ArrowFiles
# doubleScores <- addDoubletScores(input = ArrowFiles,
#                                 k = 10 #Refers to how many cells near a "pseudo-doublet" to count.
#                                 )
#
#
projHeme1 <- ArchRProject(
  ArrowFiles = ArrowFiles,
  outputDirectory = "raw-ArchR", showLogo = FALSE,
  copyArrows = FALSE #This is recommened so that if you modify the Arrow files you have an original copy for later usage.
)
projHeme1

# projHeme1 <- ArchRProject(
#   ArrowFiles = ArrowFiles,
#   outputDirectory = "rawdata",
#   copyArrows = FALSE #This is recommened so that if you modify the Arrow files you have an original copy for later usage.
# )
saveArchRProject(ArchRProj = projHeme1, outputDirectory = "Save-ProjHeme1", load = FALSE)
#
# projHeme1 <- readRDS("Save-ProjHeme1/Save-ArchR-Project.rds")
#
# clusters <- c()
# finalcells <- c()
# for (s in getSampleNames(ArchRProj = projHeme1)) {
#   dat <- read.csv(paste("cluster_names/",s,sep="",collapse=""))
#   idxSample <- BiocGenerics::which(projHeme1$Sample %in% s)
#   cellsSample <- projHeme1$cellNames[idxSample]
#   for (c in cellsSample){
#     idx = match(c,dat$cell)
#     clusters <- append(clusters, dat$cluster[idx])
#     finalcells <- append(finalcells, c)
#   }
#   cat("Done ",s,"\n")
# }
# # projHeme1 <- addCellColData(ArchRProj = projHeme1, data = clusters,
# #       cells = finalcells, name = "Cluster")
#
# # projHeme1 <- addGroupCoverages(ArchRProj = projHeme1, groupBy = "Cluster")
# # pathToMacs2 <- findMacs2()
# # projHeme1 <- addReproduciblePeakSet(
# #     ArchRProj = projHeme1,
# #     groupBy = "Cluster",
# #     peaksPerCell = 10000,
# #     pathToMacs2 = pathToMacs2
# # )
# # projHeme1 <- saveArchRProject(ArchRProj = projHeme1, outputDirectory = "Save-ProjHeme1", load = TRUE)
#
# # gr <- getPeakSet(projHeme1)
# # print(gr)
# # #Creating Label file
# # df <- data.frame(score=score(gr))
# # write.table(df, file="dendritic_peaks_labels", quote=F, row.names=F, col.names=F)
#
# #Creating bed file
# # df <- data.frame(seqnames=seqnames(gr),
# #   starts=start(gr)-1,
# #   ends=end(gr),
# #   names=c(rep(".", length(gr))),
# #   scores=c(rep(".", length(gr))),
# #   strands=strand(gr))
# # write.table(df, file="dendritic_peaks.bed", quote=F, sep="\t", row.names=F, col.names=F)
