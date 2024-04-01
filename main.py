from Bio import SeqIO
from collections import defaultdict
import numpy as np

def generate_kmers(sequence, k):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def build_kmer_index(sequences, k):
    kmer_index = defaultdict(int)
    for seq in sequences:
        for kmer in generate_kmers(seq, k):
            kmer_index[kmer] = 0
    return {kmer: idx for idx, kmer in enumerate(kmer_index.keys())}

def vectorize_sequence(sequence, kmer_index, k):
    vector = np.zeros(len(kmer_index))
    for kmer in generate_kmers(sequence, k):
        if kmer in kmer_index:
            vector[kmer_index[kmer]] += 1
    return vector

def main(fasta_file, k):
    sequences = [str(record.seq).upper() for record in SeqIO.parse(fasta_file, "fasta")]
    kmer_index = build_kmer_index(sequences, k)
    vectors = np.array([vectorize_sequence(seq, kmer_index, k) for seq in sequences])
    for kmer, idx in kmer_index.items():
        print(f"{kmer}: {vectors[0, idx]}")
    return vectors


fasta_file = "fasta/site.02.subj.0001.lab.2014222001.iso.1.fasta" 
k = 3 # k-mer length here
sequence_vectors = main(fasta_file, k)