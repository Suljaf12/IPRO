from CNNModel import *

if __name__ == "__main__":
    fasta_files = [
        "fasta/site.02.subj.0001.lab.2014222001.iso.1.fasta",
        "fasta/site.02.subj.0002.lab.2014222005.iso.1.fasta",
        "fasta/site.02.subj.0004.lab.2014222010.iso.1.fasta",
        "fasta/site.02.subj.0005.lab.2014222011.iso.1.fasta",
        "fasta/site.02.subj.0006.lab.2014222013.iso.1.fasta",
        "fasta/site.02.subj.0007.lab.2014222016.iso.1.fasta",
        "fasta/site.02.subj.0008.lab.2014222017.iso.1.fasta",
        "fasta/site.02.subj.0009.lab.2014222037.iso.1.fasta",
        "fasta/site.02.subj.0010.lab.2014222040.iso.1.fasta",
        "fasta/site.02.subj.0011.lab.2014222046.iso.1.fasta"
    ]

    # Load the data
    X, y = load_data(fasta_files)

    #print(X)

    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Print the shapes of the resulting sets
    print("Training set:", X_train.shape, y_train.shape)
    print("Validation set:", X_val.shape, y_val.shape)
    print("Test set:", X_test.shape, y_test.shape)

    # Build the CNN model
    input_shape = X_train.shape[1:]
    model = build_cnn_model(input_shape)

    # Compile the model
    compile_model(model)

    # Train the model
    history = train_model(model, X_train, y_train, X_val, y_val)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

    model.summary()


