import math
import os
import sys
import itertools
import numpy as np
import tensorflow as tf
from ucscgenome import Genome
import csv
import pybedtools

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
        chromosome = seq[0]
        #extracting chromosome number from chromosomes of form chr1_KI270710v1_random
        # label = chromosome.split('chr')[1].split('_')[0]
        label_list.append(labelseq)
        start = int(seq[1])
        end = int(seq[2])
        sequence = genomeObject[chromosome][start:end]
        encodedSequence = oneHotEncodeSequence(sequence)
        final_sequences.append(encodedSequence)
    final_sequences = np.stack(final_sequences, axis=0)
    label_list = np.stack(label_list,axis=0)
    # return final_sequences,label_list
    np.save(sequence_file, final_sequences)
    np.save(label_file, label_list)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _progress(curr, end, message):
    sys.stdout.write('\r>> %s %.1f%%' % (message, float(curr) / float(end) * 100.0))
    sys.stdout.flush()

def convert_train(train_seqs, train_labels):
    num_examples, _, _ = train_seqs.shape
    assert(num_examples == train_labels.shape[0])

    filename = 'train.tfrecord'
    num_examples_per_shard = int(math.ceil(float(num_examples) / FLAGS.num_train_shards))
    for shard_id in range(FLAGS.num_train_shards):
        output_filename = '%s-%.5d-of-%.5d' % (filename, shard_id, FLAGS.num_train_shards)
        output_file = os.path.join(FLAGS.data_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)
        start_idx = shard_id * num_examples_per_shard
        for i in range(start_idx, min(num_examples, start_idx + num_examples_per_shard)):
            seq = train_seqs[i,:, :]
            label = train_labels[i,:]
            example = tf.train.Example(features=tf.train.Features(feature={
                'seq_raw': _bytes_feature(seq.tostring()),
                'label_raw': _bytes_feature(label.tostring())}))
            writer.write(example.SerializeToString())
            _progress(i + 1, num_examples, 'Writing %s' % filename)
        writer.close()
    print

def convert_val_test(seqs, labels, split):
    num_examples, _, _ = seqs.shape
    assert(num_examples == labels.shape[0])

    filename = os.path.join(FLAGS.data_dir, split + '.tfrecord')
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(num_examples):
        seq = seqs[i, :, :].T
        label = labels[i, :]
        example = tf.train.Example(features=tf.train.Features(feature={
            'seq_raw': _bytes_feature(seq.tostring()),
            'label_raw': _bytes_feature(label.tostring())}))
        writer.write(example.SerializeToString())
        _progress(i + 1, num_examples, filename)
    writer.close()

    np.save(os.path.join(FLAGS.data_dir, split + '.npy'), labels)
    print

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
pybedtools.BedTool(list(peak_map.keys())).saveas('combined.bed')

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
for item in peak_map:
    chromosome = item[0]
    if chromosome.startswith('chr8') or chromosome.startswith('chr9'):
        testSet.append((item,peak_map[item]))
    elif chromosome.startswith('chr4'):
        validationSet.append((item,peak_map[item]))
    else:
        # if(checkAllImb(item,all_imb_list)):
        trainingSet.append((item,peak_map[item]))

print(len(trainingSet),len(validationSet),len(testSet))
# Generate one hot encoding and labels
print('Started one hot encoding')
genome_dir = '/home/eramamur/resources/genomes/hg38'
genomeObject = Genome('hg38', cache_dir=genome_dir, use_web=False)
generateOneHotEncodedSequences(trainingSet,'./trainInput','./trainLabels')
# print('Begin sharding')
# convert_train(seq,labels)
generateOneHotEncodedSequences(validationSet,'./validationInput','./validationLabels')
# convert_val_test(seq,labels,'val')
generateOneHotEncodedSequences(testSet,'./testInput','./testLabels')
# convert_val_test(seq,labels,'test')
