# RNN-Based IMDB Sentiment Analysis

## Project Overview

This project uses Natural Language Processing (NLP) and a Recurrent Neural Network (RNN) built with PyTorch to classify IMDB movie reviews by sentiment. The dataset contains movie reviews labelled as **positive** or **negative**.

The notebook demonstrates a text-processing pipeline, sentiment-label encoding, TF-IDF feature extraction, model training, and evaluation on a held-out test set.

## Objectives

- Prepare raw movie reviews for machine learning.
- Convert text into numerical features using TF-IDF.
- Encode sentiment labels into numeric values.
- Build and train an RNN model using PyTorch.
- Evaluate predictions on test data.

## Technologies Used

- Python
- Pandas
- Regular expressions (`re`)
- NLTK
- Scikit-learn
- PyTorch
- Jupyter Notebook

## Dataset

The notebook expects a CSV file named `IMDB Dataset.csv` in the working directory. It should contain these columns:

- `review` — the movie review text
- `sentiment` — the review label, such as `positive` or `negative`

The notebook checks for missing values and removes duplicate rows. Make sure you have the right to use the dataset and follow its terms.

## Workflow

### 1. Load and inspect the data
Load the CSV with Pandas, inspect its shape and sample rows, check missing values, and remove duplicate records.

### 2. Preprocess the text
The notebook applies these cleaning steps:

1. Convert reviews to lowercase.
2. Remove URLs.
3. Remove punctuation and other non-alphanumeric characters.
4. Remove HTML tags.
5. Remove English stopwords.
6. Apply Porter stemming to reduce words to their stems.

NLTK resources used in the notebook include `punkt`, `punkt_tab`, and `stopwords`.

### 3. Encode sentiment labels
`LabelEncoder` converts the sentiment labels into numeric classes.

### 4. Convert text to TF-IDF features
`TfidfVectorizer(max_features=5000)` transforms the cleaned reviews into numerical feature vectors using up to 5,000 features.

### 5. Split the dataset
The data is split into training and testing sets with an 80/20 ratio and `random_state=42`.

### 6. Create PyTorch data loaders
The feature matrices and labels are converted to PyTorch tensors and loaded in batches of 64.

### 7. Build and train the RNN
The model uses a PyTorch `nn.RNN` layer with a hidden size of 128 and a fully connected output layer. Training runs for 10 epochs with the Adam optimizer and binary cross-entropy loss.

### 8. Evaluate the model
The notebook predicts sentiment labels for the test set using a threshold of 0.5 and reports classification accuracy.

## How to Run

1. Clone or download this repository.
2. Place `IMDB Dataset.csv` in the same directory as the notebook.
3. Install the dependencies:

   ```bash
   pip install pandas nltk scikit-learn torch jupyter
   ```

4. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open `RNN_IMDB_SENTIMENT_ANALYSIS.ipynb` and run the cells in order.

The notebook downloads the required NLTK resources in its preprocessing section.

## Important Implementation Note

In the current notebook, TF-IDF produces a feature vector for each review, and the model input is expanded with `unsqueeze(1)`. This gives the RNN a sequence length of **one**, so it does not process the review as a word-by-word sequence. This is the implementation currently present in the notebook; a sequence-based RNN would typically use tokenized word sequences and embeddings.

Also, the notebook uses `nn.BCELoss()` after applying a sigmoid to the model output. The encoded labels should be checked to ensure the positive class is represented consistently with the chosen probability interpretation.

## Results

The notebook prints the training loss for each epoch and the final test accuracy when run. Add your actual output here after running the notebook:

- **Test accuracy:** _Add the measured accuracy here_
- **Epochs:** 10
- **Batch size:** 64
- **TF-IDF maximum features:** 5,000
- **RNN hidden size:** 128

No accuracy value is stated here because it should be taken from the actual notebook run.

## Future Improvements

- Use tokenized word sequences and an embedding layer so the RNN can process word order.
- Compare RNN results with LSTM or GRU models.
- Report precision, recall, F1-score, and a confusion matrix alongside accuracy.
- Improve preprocessing and evaluate its effect on model performance.
- Save the trained model and preprocessing components for reuse.

## Repository Structure

```text
.
├── RNN_IMDB_SENTIMENT_ANALYSIS.ipynb
├── IMDB Dataset.csv
└── README.md
```

## License

Add a license if you plan to distribute this project publicly. Check the dataset's license and terms separately before sharing the dataset.

---

**Project:** RNN-Based IMDB Sentiment Analysis  
**Task:** Binary sentiment classification of movie reviews
