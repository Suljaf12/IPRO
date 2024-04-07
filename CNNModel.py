import csv
import numpy as np
from Bio import SeqIO
from sklearn.model_selection import train_test_split
#from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.python.keras.models import Sequential


####################################################################
############### Old/Modified Functions from main.py#################
####################################################################

def generate_normalized_kmer_frequencies(sequence, k_range):
    """
    Generate normalized k-mer frequencies for a given sequence.

    Parameters:
        sequence (str): Input DNA sequence.
        k_range (range): Range of k-mer lengths.

    Returns:
        dict: Dictionary containing normalized k-mer frequencies.
    """
    kmer_counts = {}
    total_kmers = 0

    # Generate and count k-mers
    for k in k_range:
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i + k]
            kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1
            total_kmers += 1

    # Normalize k-mer frequencies
    for kmer in kmer_counts:
        kmer_counts[kmer] /= total_kmers

    return kmer_counts


def build_global_kmer_index(sequences, k_range):
    """
    Build a global k-mer index for a list of DNA sequences.

    Parameters:
        sequences (list): List of DNA sequences.
        k_range (range): Range of k-mer lengths.

    Returns:
        dict: Dictionary containing global k-mer index.
    """
    global_kmer_index = {}
    for sequence in sequences:
        for k in k_range:
            for i in range(len(sequence) - k + 1):
                kmer = sequence[i:i + k]
                if kmer not in global_kmer_index:
                    global_kmer_index[kmer] = len(global_kmer_index)
    return global_kmer_index


def vectorize_sequence(sequence, global_kmer_index, k_range):
    """
    Vectorize a DNA sequence using a global k-mer index.

    Parameters:
        sequence (str): Input DNA sequence.
        global_kmer_index (dict): Global k-mer index.
        k_range (range): Range of k-mer lengths.

    Returns:
        numpy.ndarray: Vector representation of the input sequence.
    """
    vector = np.zeros(len(global_kmer_index))
    kmer_frequencies = generate_normalized_kmer_frequencies(sequence, k_range)

    for kmer, frequency in kmer_frequencies.items():
        if kmer in global_kmer_index:
            vector[global_kmer_index[kmer]] = frequency

    return vector


def process_fasta_files(fasta_files):
    """
    Process FASTA files to extract DNA sequences.

    Parameters:
        fasta_files (list): List of paths to FASTA files.

    Returns:
        list: List of DNA sequences.
    """
    sequences = []
    for fasta_file in fasta_files:
        sequences.extend([str(record.seq).upper() for record in SeqIO.parse(fasta_file, "fasta")])
    return sequences


def prepare_sequence_data_for_cnn(fasta_files):
    """
    Prepare sequence data for input to a Convolutional Neural Network (CNN).

    Parameters:
        fasta_files (list): List of paths to FASTA files.

    Returns:
        numpy.ndarray: Array of sequence vectors formatted for CNN input.
    """
    k_range = range(1, 7)  # K-mers from 1 to 6
    sequences = process_fasta_files(fasta_files)
    global_kmer_index = build_global_kmer_index(sequences, k_range)

    # Vectorize sequences
    vectors = np.array([vectorize_sequence(seq, global_kmer_index, k_range) for seq in sequences])

    # Reshape for CNN input
    vectors = vectors.reshape(vectors.shape[0], vectors.shape[1], 1)

    return vectors

def save_vectors_to_csv(sequence_vectors, output_file):
    """
    Save sequence vectors to a CSV file.

    Parameters:
        sequence_vectors (numpy.ndarray): Array of sequence vectors.
        output_file (str): Path to the output CSV file.

    Returns:
        None
    """
    # Reshape the sequence vectors for CSV writing
    reshaped_vectors = sequence_vectors.reshape(sequence_vectors.shape[0], -1)

    # Save to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(reshaped_vectors)

####################################################################
################# New Functions for Modal###########################
####################################################################
def load_data(fasta_files):
    """
    Load and preprocess data from FASTA files.

    Parameters:
        fasta_files (list): List of paths to FASTA files.

    Returns:
        numpy.ndarray: Array of input features.
        numpy.ndarray: Array of target labels.
    """
    # Process the FASTA files to obtain sequence vectors
    sequence_vectors = prepare_sequence_data_for_cnn(fasta_files)

    # Save vectors to csv
    output_file = 'output.csv'
    save_vectors_to_csv(sequence_vectors, output_file)

    # Generate binary labels for demonstration
    labels = np.random.randint(2, size=(sequence_vectors.shape[0],))

    return sequence_vectors, labels


def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split the data into training, validation, and test sets.

    Parameters:
        X (numpy.ndarray): Array of input features.
        y (numpy.ndarray): Array of target labels.
        test_size (float): Proportion of the dataset to include in the test split (default is 0.2).
        val_size (float): Proportion of the training set to include in the validation split (default is 0.2).
        random_state (int): Random seed for reproducibility (default is 42).

    Returns:
        Tuple: Tuple containing training, validation, and test sets (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test

def build_cnn_model(input_shape):
    """
    Build a Convolutional Neural Network (CNN) model.

    Parameters:
        input_shape (tuple): Shape of the input data (e.g., (sequence_length, 1)).

    Returns:
        tensorflow.keras.models.Sequential: Compiled CNN model.
    """
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(units=64, activation='relu'),
        Dense(units=1, activation='sigmoid')
    ])
    return model

def compile_model(model):
    """
    Compile the specified CNN model.

    Parameters:
        model (tensorflow.keras.models.Sequential): CNN model to compile.

    Returns:
        None
    """
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    """
    Train the specified CNN model.

    Parameters:
        model (tensorflow.keras.models.Sequential): CNN model to train.
        X_train (numpy.ndarray): Training data features.
        y_train (numpy.ndarray): Training data labels.
        X_val (numpy.ndarray): Validation data features.
        y_val (numpy.ndarray): Validation data labels.
        epochs (int): Number of epochs for training (default is 10).
        batch_size (int): Batch size for training (default is 32).

    Returns:
        tensorflow.python.keras.callbacks.History: Training history.
    """
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
    return history

