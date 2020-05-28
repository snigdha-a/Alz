import math
import os
import sys
import itertools
import numpy as np
import tensorflow as tf
from ucscgenome import Genome
import csv
import pybedtools
from Bio import SeqIO
import pyfasta
import random
import glob

def oneHotEncodeSequence(sequence):
    oneHotDimension = (len(sequence), 4)
    dnaAlphabet = {"A":0, "G":1, "C":2, "T":3}
    one_hot_encoded_sequence = np.zeros(oneHotDimension, dtype=np.int)
    for i, nucleotide in enumerate(sequence):
        if nucleotide.upper() in dnaAlphabet:
            index = dnaAlphabet[nucleotide.upper()]
            one_hot_encoded_sequence[i][index] = 1
    return one_hot_encoded_sequence

def labelOneHotEncoded(label):
    one_hot_encoded_label = np.zeros(8, dtype=np.int)
    one_hot_encoded_label[label-1] = 1
    return one_hot_encoded_label

def generateOneHotEncodedSequences(peak_list, sequence_file, label_file):
    label_list = []
    final_sequences = []
    for (seq,labelseq) in peak_list:
        # Normal sequence encoding and appending
        chromosome = seq[0]
        #extracting chromosome number from chromosomes of form chr1_KI270710v1_random
        # label = chromosome.split('chr')[1].split('_')[0]
        label_list.append(labelseq)
        start = int(seq[1])
        end = int(seq[2])
        sequence = genomeObject[chromosome][start:end]
        encodedSequence = oneHotEncodeSequence(sequence)
        final_sequences.append(encodedSequence)
        '''Create fasta for each entry and then generate reverse
        complement and store back in peak_map'''
        # a = pybedtools.BedTool([ seq ]).sequence()
        # a.save_seqs('temp.fa')
        ofile = open("temp.fa", "w")
        ofile.write(">" + str(seq) + "\n" +sequence + "\n")
        ofile.close()
        records = [rec.reverse_complement(id="rc_"+rec.id, description = "reverse complement")
        for rec in SeqIO.parse("temp.fa", "fasta")]
        SeqIO.write(records, "temp.fa", "fasta")
        f = pyfasta.Fasta("temp.fa")
        for header in f.keys():
            sequence = str(f[header])
            encodedSequence = oneHotEncodeSequence(sequence)
            label_list.append(labelseq)
            final_sequences.append(encodedSequence)
    final_sequences = np.stack(final_sequences, axis=0)
    label_list = np.stack(label_list,axis=0)
    # return final_sequences,label_list
    np.save(sequence_file, final_sequences)
    np.save(label_file, label_list)
    # to remove the temp files created above
    for filename in glob.glob("temp*"):
        os.remove(filename)



def checkAllImb(read,all_imb_list):
    chromosome = read[0]
    start = int(read[1])
    end = int(read[2])
    for item in all_imb_list:
        if item['chr']==chromosome:
            if int(item['pos'])<=end or int(item['pos'])>=start:
                return False
    return True

#read chromsizes and store into map
chrom_size_dict = {}
with open("chromsizes") as f:
    for line in f:
        row = line.strip().split()
        chrom_size_dict[row[0]] = row[1]

#Convert all peaks to size 500 by binning
# path = "/projects/pfenninggroup/machineLearningForComputationalBiology/gwasEnrichments/foreground/blood_scATAC-seq/reduced_clusters_snigdha/"
path = "/projects/pfenninggroup/machineLearningForComputationalBiology/gwasEnrichments/foreground/blood_scATAC-seq/clusters/"
#contains [chromosome,start,end] as key and 00010100 as values
peak_map={}
for cluster in range(1,9):
    with open(path+str(cluster)+"_hg38.bed") as f:
        for line in f:
            line = line.strip().split()
            start = int(line[1])
            end = int(line[2])
            peak_length = end-start+1
            desired_peak_size = 500
            #ignoring super enhancer peaks and too small peaks
            if peak_length < 1000 and peak_length >= 250:
                cur_chrom = line[0]
                cur_chrom_size = int(chrom_size_dict[cur_chrom])
                #peak smaller than 500
                #Throw away
                # if peak_length < desired_peak_size:
                #     end = end + math.ceil((500-peak_length)/2)
                #     start = start - math.floor((500-peak_length)/2)
                #     #check if coordinates are outside chromosome size
                #     if end > cur_chrom_size:
                #         #shift peak towards start
                #         start = start - (cur_chrom_size - end)
                #         end = cur_chrom_size
                #     elif start < 0:
                #         end = end + (0 - start) #shift peak towards end
                #         start = 0
                #     line[1] = start
                #     line[2] = end
                #     line.append(i)
                #     content.append(line)
                # elif peak_length >= desired_peak_size: #binning
                stride = 100
                binList = []
                #bins start considering the current peak start as midpoint
                #and end considering current peak end as midpoint
                for bin_start in range(start - int(desired_peak_size/2),
                end - int(desired_peak_size/2),stride):
                    bin_end = bin_start + desired_peak_size - 1
                    if bin_start > 0 and bin_end < cur_chrom_size:
                        new_peak = (cur_chrom,bin_start,bin_end)
                        #check if current peak already exists in peakMap
                        if new_peak not in peak_map:
                            peak_map[new_peak] = labelOneHotEncoded(cluster)
                        else: #set 1 in the corresponding label position
                            peak_map[new_peak][cluster-1] = 1

#remove duplicates
# content.sort()
# peaks = list(k for k,_ in itertools.groupby(content))
# print(len(peaks))
# peaks = content

#Storing all training data into 1 big bed file
# pybedtools.BedTool(list(peak_map.keys())).saveas('combined.bed')

#check if correct sized peaks generated
for item in peak_map:
    if item[2]-item[1]+1 != 500: #including first element of peak
        print(item)

#Create training, test and validation data
trainingSet=[]
testSet=[]
validationSet=[]
all_imb_list=[]
#Removing allelic imbalance data SNPs from train,validation and tesdt data
# with open('Supplementary_data_5_aggregated_gwas_asc_da_de.txt') as mydata:
#      data_reader = csv.DictReader(mydata, delimiter='\t')
#      for entry in data_reader:
#          all_imb_list.append(dict(entry))

''' Multi label scenario'''
# for item in peak_map:
#     chromosome = item[0]
#     if chromosome.startswith('chr8') or chromosome.startswith('chr9'):
#         testSet.append((item,peak_map[item]))
#     elif chromosome.startswith('chr4'):
#         validationSet.append((item,peak_map[item]))
#     else:
#         # if(checkAllImb(item,all_imb_list)):
#         trainingSet.append((item,peak_map[item]))

''' This flow is for single label where peaks from other clusters are taken to
get negatives'''
# 1 = dendritic, 2 = monocytes, 3 = B cells
# label_index = 1
# # extract the positives from the big peak_map
positiveLabel = np.ones(1, dtype=np.int)
negative_label = np.zeros(1, dtype=np.int)
# # dictionary with chrom as key and star,stop pairs as value to check for overlap
# allpositives = {}
# for item in peak_map:
#     chromosome = item[0]
#     if peak_map[item][label_index] == 1:
#         if chromosome in allpositives:
#             allpositives[chromosome].append((item[1],item[2]))
#         else:
#             allpositives[chromosome] = [(item[1],item[2])]
#         if chromosome.startswith('chr8') or chromosome.startswith('chr9'):
#             testSet.append((item, positiveLabel))
#         elif chromosome.startswith('chr4'):
#             validationSet.append((item, positiveLabel))
#         else:
#             # if(checkAllImb(item,all_imb_list)):
#             trainingSet.append((item, positiveLabel))
# # Get number of positives and add 2 times the number of negatives
# positives = len(peak_map)
# positives *= 2
# # To ensure the negatives are random and not belonging to a particular grouping
# # Shuffle the keys in dictionary
# keys =  list(peak_map.keys())
# random.shuffle(keys)
# for item in keys:
#     chromosome = item[0]
#     overlap = False
#     for interval in allpositives.get(chromosome,[]):
#         if ((interval[0] < item[1] and item[1] < interval[1]) or
#         (interval[0] < item[2] and item[2] < interval[1])):
#             overlap = True
#             break
#     if overlap==False:
#         if peak_map[item][label_index] == 0 and positives > 0:
#             if chromosome.startswith('chr8') or chromosome.startswith('chr9'):
#                 testSet.append((item, negative_label))
#             elif chromosome.startswith('chr4'):
#                 validationSet.append((item, negative_label))
#             else:
#                 trainingSet.append((item,negative_label))
#                 positives -= 1

''' This flow is to create negative set that is GC matched negative set and not
 use differential peaks defined negatives'''

# 1 = dendritic, 2 = monocytes, 3 = B cells
# label_index = 1
# # # Create positive bed file to feed into R Rscript
# posSet = []
# for item in peak_map:
#     chromosome = item[0]
#     if peak_map[item][label_index] == 1:
#         posSet.append(item)
#         if chromosome.startswith('chr8') or chromosome.startswith('chr9'):
#             testSet.append((item, positiveLabel))
#         elif chromosome.startswith('chr4'):
#             validationSet.append((item, positiveLabel))
#         else:
#             trainingSet.append((item, positiveLabel))
# # pybedtools.BedTool(posSet).saveas('label_'+str(label_index)+'.bed')
# # os.system("Rscript nullSet.R label_"+ str(label_index))
# with open("label_"+str(label_index)+"-neg.bed")as f:
#     for line in f:
#         negPeak = line.strip().split()
#         chromosome = negPeak[0]
#         # TODO: DOnt know why genNullSeqs generates 498 length negatives.
#         item = (chromosome, negPeak[1], int(negPeak[2])+1)
#         if chromosome.startswith('chr8') or chromosome.startswith('chr9'):
#             testSet.append((item, negative_label))
#         elif chromosome.startswith('chr4'):
#             validationSet.append((item, negative_label))
#         else:
#             trainingSet.append((item,negative_label))

''' Old data - no directionality - dendritic cells - bed files already present'''
with open("combined2GC-neg.bed") as f:
    for line in f:
        negPeak = line.strip().split()
        chromosome = negPeak[0]
        # TODO: DOnt know why genNullSeqs generates 498 length negatives.
        item = (chromosome, negPeak[1], int(negPeak[2])+1)
        if chromosome.startswith('chr8') or chromosome.startswith('chr9'):
            testSet.append((item, negative_label))
        elif chromosome.startswith('chr4'):
            validationSet.append((item, negative_label))
        else:
            trainingSet.append((item,negative_label))

with open("combined2pos.bed") as f:
    for line in f:
        peak = line.strip().split()
        chromosome = peak[0]
        item = (chromosome,peak[1],peak[2])
        if chromosome.startswith('chr8') or chromosome.startswith('chr9'):
            testSet.append((item, positiveLabel))
        elif chromosome.startswith('chr4'):
            validationSet.append((item, positiveLabel))
        else:
            trainingSet.append((item, positiveLabel))

print(len(trainingSet),len(validationSet),len(testSet))
# Generate one hot encoding and labels
print('Started one hot encoding')
genome_dir = '/home/eramamur/resources/genomes/hg38'
genomeObject = Genome('hg38', cache_dir=genome_dir, use_web=False)
generateOneHotEncodedSequences(trainingSet,'./trainInput','./trainLabels')
generateOneHotEncodedSequences(validationSet,'./validationInput','./validationLabels')
generateOneHotEncodedSequences(testSet,'./testInput','./testLabels')
