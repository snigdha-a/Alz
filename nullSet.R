#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
library(gkmSVM)
library(BSgenome.Hsapiens.UCSC.hg38.masked)
genome <- BSgenome.Hsapiens.UCSC.hg38.masked
genNullSeqs(paste(args[1],'.bed',sep=""), #"combined2pos.bed",
  genome=genome,
  outputBedFN = paste(args[1],'-neg.bed',sep=""),#'combined2GC-neg.bed',
  outputPosFastaFN = paste(args[1],'.fa',sep=""),#'combined2pos.fa',
  outputNegFastaFN = paste(args[1],'-neg.fa',sep=""),#'neg2GC-combined.fa',
  length_match_tol = 0, # want exact same length negatives
  xfold = 2, # twice the number of negatives
	nMaxTrials=300)
