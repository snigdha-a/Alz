import argparse
import numpy as np
from ucscgenome import Genome
import pandas as pd

def oneHotEncodeSequence(sequence):
    oneHotDimension = (len(sequence), 4)
    dnaAlphabet = {"A":0, "G":1, "C":2, "T":3}
    one_hot_encoded_sequence = np.zeros(oneHotDimension, dtype=np.int)
    for i, nucleotide in enumerate(sequence):
        if nucleotide.upper() in dnaAlphabet:
            index = dnaAlphabet[nucleotide.upper()]
            one_hot_encoded_sequence[i][index] = 1
    return one_hot_encoded_sequence

def getOneHotEncodedSequences(bedInput, xOutput, xAlternateOutput, leftWindow, rightWindow, genomeObject):
    finalReferenceSequences = []
    finalAlternateSequences = []
    final_xlsx = []
    with open(bedInput, 'r') as f:
        for line in f:
            curInterval = line.strip().split("\t")
            ref = curInterval[3]
            alt_arr = curInterval[4].split(',')
            chrom = curInterval[5].split(':')[0]
            positions = curInterval[5].split(':')[1].split('-')
            start = int(positions[0])
            end = int(positions[1])

            if len(ref)==1:
            # Note: genomeobject returns seq from 1 to 100 if given Genome[chr][0:100], also -1 because want 500 length seq
                referenceSequence  = genomeObject[chrom][start - leftWindow:start] + ref + genomeObject[chrom][end: end+rightWindow-1]
            else:
                centre = end - int(len(ref)/2)
                # -1 accounts for starting position like above
                referenceSequence  = genomeObject[chrom][centre - leftWindow -1:end-len(ref)] + ref + genomeObject[chrom][end: centre+rightWindow-1]
            assert(len(referenceSequence)==leftWindow+rightWindow)

            for alt in alt_arr:
                if len(alt)==1:
                    alternateSequence = genomeObject[chrom][start-leftWindow:start] + alt + genomeObject[chrom][end: end+rightWindow-1]
                else:
                    centre = end - int(len(alt)/2)
                    alternateSequence = genomeObject[chrom][centre-leftWindow-1:end-len(alt)] + alt + genomeObject[chrom][end: centre+rightWindow-1]
                assert(len(alternateSequence)==leftWindow+rightWindow)

                final_xlsx.append(np.append(curInterval[:4],[alt,curInterval[5],referenceSequence,alternateSequence]))
                # encodedReferenceSequence = oneHotEncodeSequence(referenceSequence)
                # encodedAlternateSequence = oneHotEncodeSequence(alternateSequence)
                # finalReferenceSequences.append(encodedReferenceSequence)
                # finalAlternateSequences.append(encodedAlternateSequence)

    final_xlsx = np.stack(final_xlsx,axis=0)
    print(final_xlsx.shape)
    pd.DataFrame(final_xlsx).to_excel('combined.xlsx', header=False, index=False)
    # finalReferenceSequences = np.stack(finalReferenceSequences, axis=0)
    # finalAlternateSequences = np.stack(finalAlternateSequences, axis=0)
    #
    # print(finalReferenceSequences.shape)
    # print(finalAlternateSequences.shape)
    # np.save(xOutput, finalReferenceSequences)
    # np.save(xAlternateOutput, finalAlternateSequences)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Construct numpy arrays containing one-hot encoded sequences for given snp bed file along with window size (BE CAREFUL IF WINDOWS ARE TOO BIG, SNP METADATA MAY NOT MATCH THE NUMPY ARRAYS)")
    parser.add_argument('-i', '--bed-like-input', help='bed file containing coordinates for snps, alleles, p and z scores', required=True)
    parser.add_argument('-xo', '--x-output', help = 'output numpy array of one hot encoded sequences', required=True)
    parser.add_argument('-xao', '--x-alternate-output', help = 'output numpy array of one hot encoded sequences for alternate alleles', required=True)
    parser.add_argument('-l', '--left-window', type=int, help = 'how many nucleotides to pad at the left', required=True)
    parser.add_argument('-r', '--right-window', type=int, help = 'how many nucleotides to pad at the right', required=True)
    parser.add_argument('-g', '--genome-name', help='name of the genome', required=True)
    parser.add_argument('-d', '--genome-dir', help='local path to genomes (MUST BE in 2-bit format)', required=True)

    args = parser.parse_args()
    bedInput = args.bed_like_input
    xOutput = args.x_output
    xAlternateOutput = args.x_alternate_output
    leftWindow = args.left_window
    rightWindow = args.right_window
    genomeName = args.genome_name
    genomeDir = args.genome_dir
    genomeObject = Genome(genomeName, cache_dir=genomeDir, use_web=False)
    getOneHotEncodedSequences(bedInput, xOutput, xAlternateOutput, leftWindow, rightWindow, genomeObject)
