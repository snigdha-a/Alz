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
library(ArchR)
# ArchR::installExtraPackages()

#post installation
addArchRThreads(threads = 4)
addArchRGenome("hg19")
# ArrowFiles <- createArrowFiles(
#   inputFiles = c("GSM3722039_Dendritic_all_cells_fragments.tsv.gz"),
#   sampleNames = c("dendritic"),
#   verbose=TRUE#,
#   # filterTSS = 4, #Dont set this too high because you can always increase later
#   # filterFrags = 1000,
#   # addTileMat = TRUE,
#   # addGeneScoreMat = TRUE
# )
ArrowFiles <- c("dendritic.arrow")
projHeme1 <- ArchRProject(
  ArrowFiles = ArrowFiles,
  outputDirectory = "rawdata",
  copyArrows = TRUE #This is recommened so that if you modify the Arrow files you have an original copy for later usage.
)
head(projHeme1$cellNames)
