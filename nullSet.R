library(gkmSVM)
library(BSgenome.Hsapiens.UCSC.hg38.masked)
genome <- BSgenome.Hsapiens.UCSC.hg38.masked
genNullSeqs("combined.bed",
  genome=genome,
  outputBedFN = 'negSet.bed',
  outputPosFastaFN = 'posSet.fa',
  outputNegFastaFN = 'negSet.fa',
	nMaxTrials=200)
