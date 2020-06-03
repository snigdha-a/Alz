import re
from scipy.io import mmread
from itertools import *
import os

def generateSignals():
    # load peaks.txt into a list
    peaks=[]
    path = '/projects/pfenninggroup/machineLearningForComputationalBiology/eramamur_stuff/satpathy_blood_scatac/'
    open_file = open(path+'GSE129785_scATAC-Hematopoiesis-All.peaks.txt')
    for line in open_file:
        peaks.append(line.rstrip('\n'))
    print("Done with peaks list")

    # get indices of cluster specific cells
    cluster_indices = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
    open_file = open(path+'GSE129785_scATAC-Hematopoiesis-All.cell_barcodes.txt')
    for i,line in enumerate(open_file):
        columns = line.split()
        cluster = columns[2]
        index = i - 1 # ignoring first line in barcodes.txt
        if cluster in ['Cluster'+str(a) for a in range(1,10)]:
            cluster_indices[1].append(index)
        elif cluster in ['Cluster'+str(a) for a in range(10,12)]:
            cluster_indices[2].append(index)
        elif cluster in ['Cluster'+str(a) for a in range(12,14)]:
            cluster_indices[3].append(index)
        elif cluster in ['Cluster'+str(a) for a in range(14,17)]:
            cluster_indices[4].append(index)
        elif cluster == 'Cluster17':
            cluster_indices[5].append(index)
        elif cluster in ['Cluster'+str(a) for a in range(18,21)]:
            cluster_indices[6].append(index)
        elif cluster in ['Cluster'+str(a) for a in range(21,26)]:
            cluster_indices[7].append(index)
        elif cluster in ['Cluster'+str(a) for a in range(26,32)]:
            cluster_indices[8].append(index)
    print("Done with cluster indices")

    # load sparse matrix
    peak_by_cell_matrix = mmread(path+'GSE129785_scATAC-Hematopoiesis-All.mtx')
    matrix_csr = peak_by_cell_matrix.tocsr()
    # go over each peak in every bed files
    path = "/projects/pfenninggroup/machineLearningForComputationalBiology/gwasEnrichments/foreground/blood_scATAC-seq/reduced_clusters_snigdha/"
    for cluster in range(1,9):
        label_file = open(str(cluster)+'label',"w+")
        with open(path+str(cluster)+"_hg19.bed") as f:
            for line in f:
                peak = re.sub('  ','_',line.rstrip('\n'))
                matrix_row = peaks.index(peak) - 1 # ignore first row in peaks.txt
                columns = cluster_indices[cluster]
                #extract columns from matrix
                sum = matrix_csr[matrix_row,columns].sum(axis=1)
                label_file.write(str(sum)+'\n')

def extendPeaks():
    #read chromsizes and store into map
    chrom_size_dict = {}
    with open("chromsizes") as f:
        for line in f:
            row = line.strip().split()
            chrom_size_dict[row[0]] = row[1]
    #check peak_length
    path = "/projects/pfenninggroup/machineLearningForComputationalBiology/gwasEnrichments/foreground/blood_scATAC-seq/reduced_clusters_snigdha/"
    for cluster in range(1,9):
        with open(path+str(cluster)+"_hg19.bed") as f:
            for line in f:
                line = line.strip().split()
                cur_chrom = line[0]
                start = int(line[1])
                end = int(line[2])
                cur_chrom_size = int(chrom_size_dict[cur_chrom])
                #TODO

def generateOneHotEncodedSequences(peak_list, sequence_file, label_file):
    genome_dir = '/home/eramamur/resources/genomes/hg19'
    genomeObject = Genome('hg19', cache_dir=genome_dir, use_web=False)
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


def generateSingleLabelData(label_index):
    cluster = label_index + 1
    # generate gc negatives
    path = "/projects/pfenninggroup/machineLearningForComputationalBiology/gwasEnrichments/foreground/blood_scATAC-seq/reduced_clusters_snigdha/"
    os.system("Rscript nullSet.R hg19 "+ path+str(cluster)+"_hg19")
    # with open(path+str(cluster)+"_hg19"+"-neg.bed")as f1,
    # open('signal_data/'+str(cluster)+'label') as f2:
    #     for peak, label in zip(f1,f2):
    #         negPeak = peak.strip().split()
    #         chromosome = negPeak[0]
    #         # TODO: DOnt know why genNullSeqs generates 498 length negatives.
    #         item = (chromosome, negPeak[1], int(negPeak[2])+1)
    #         if chromosome.startswith('chr8') or chromosome.startswith('chr9'):
    #             testSet.append((item,label[0][0]))
    #         elif chromosome.startswith('chr4'):
    #             validationSet.append((item,label[0][0]))
    #         else:
    #             trainingSet.append((item,label[0][0]))
    # print(len(trainingSet),len(validationSet),len(testSet))
    # print('Started one hot encoding')
    # generateOneHotEncodedSequences(trainingSet,'./trainInput','./trainLabels')
    # generateOneHotEncodedSequences(validationSet,'./validationInput','./validationLabels')
    # generateOneHotEncodedSequences(testSet,'./testInput','./testLabels')

if __name__=="__main__":
    # generateSignals()

    # extendPeaks()
    # generateTrainingData() # use later for multi label scenario

    # 1 = dendritic, 2 = monocytes, 3 = B cells
    label_index = 1
    generateSingleLabelData(label_index)
