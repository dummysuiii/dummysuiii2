import pandas as pd
from sklearn.model_selection import train_test_split

# Load from local CSV
data = pd.read_csv('boston.csv')

# Split into features and labels
X = data.iloc[:, :-1].values  # All columns except the last
y = data.iloc[:, -1].values   # Last column as label

# Split into train/test
from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)




import numpy as np

# Load locally saved .npz version of Boston Housing dataset
with np.load('boston_housing.npz') as data:
    train_data = data['x_train']
    train_labels = data['y_train']
    test_data = data['x_test']
    test_labels = data['y_test']




Depending on what your offline IMDB data looks like, here’s how to adjust:

📁 If You Have It as Text Files (e.g., aclImdb/train/pos, aclImdb/train/neg)
That’s the standard IMDB dataset format from AI Stanford. You need to:

Read the text reviews from directories

Assign labels (1 for pos, 0 for neg)

Tokenize and pad the text

🔧 Example:
python
Copy
Edit
import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_imdb_data_from_dir(directory):
    texts = []
    labels = []

    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(directory, label_type)
        for fname in os.listdir(dir_name):
            if fname.endswith('.txt'):
                with open(os.path.join(dir_name, fname), encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(0 if label_type == 'neg' else 1)

    return texts, labels

# Load training and testing data
train_texts, train_labels = load_imdb_data_from_dir("aclImdb/train")
test_texts, test_labels = load_imdb_data_from_dir("aclImdb/test")

# Tokenize
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train_texts)

X_train = tokenizer.texts_to_sequences(train_texts)
X_test = tokenizer.texts_to_sequences(test_texts)

# Pad sequences
X_train = pad_sequences(X_train, maxlen=256)
X_test = pad_sequences(X_test, maxlen=256)

y_train = np.array(train_labels)
y_test = np.array(test_labels)
✅ Then Replace:
python
Copy
Edit
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
With:

python
Copy
Edit
train_data, train_labels = X_train, y_train
test_data, test_labels = X_test, y_test
