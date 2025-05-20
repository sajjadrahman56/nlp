# AgroChatbot

This README explains the code for an AgroChatbot, a simple chatbot that answers frequently asked questions (FAQs) related to agriculture. The chatbot uses a TF-IDF-based approach to provide relevant answers to user queries.

## **Code Explanation**

### 1. **Imports**

```python
import re
import emoji
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
```

* **re**: Used to remove URLs and handle regular expressions.
* **emoji**: Converts emoji into text or removes them.
* **contractions**: Expands contractions like "don't" into "do not".
* **nltk**: A package for various natural language processing tasks, such as tokenization and lemmatization.
* **spellchecker**: Used to correct spelling mistakes.
* **sklearn**: Provides utilities for feature extraction (`TfidfVectorizer`) and similarity calculations (`cosine_similarity`).
* **numpy**: Used for numerical operations such as calculating the cosine similarity between vectors.

### 2. **Slang Replacements and Emotion Mapping**

```python
slang_replacements = {
    'plz': 'please', 'thx': 'thanks', 'tnx': 'thanks', 'u': 'you', 'r': 'are',
    ...
}
```

* **slang\_replacements**: A dictionary of commonly used internet slang and their corresponding formal versions.

```python
emotion_map = {
    'angry': [':angry_face:', ':pouting_face:', ':rage:'],
    ...
}
```

* **emotion\_map**: Maps emotional states (like "angry", "happy") to their corresponding emoji representations.

---

### 3. **FAQ Data**

```python
faq_data = (
    ("What are your opening hours?", "Our agro firm is open from 8:00 AM to 6:00 PM, Monday to Saturday."),
    ...
)
```

* **faq\_data**: A tuple of questions and answers. These FAQs represent the data that the chatbot will use to respond to user queries.

### 4. **NLTK Setup**

```python
NLTK_DATA_PATH = r'D:\ML\.venv\lib\nltk_data'
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_PATH)
```

* **NLTK\_DATA\_PATH**: Specifies the location where NLTK resources (like stopwords, wordnet) are stored.
* The code checks if the path exists in NLTK's data directory and adds it if necessary.

```python
for resource in ['stopwords', 'wordnet']:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, download_dir=NLTK_DATA_PATH)
```

* Downloads necessary resources (stopwords, wordnet) if they are not already available.

### 5. **Text Preprocessing Functions**

```python
def remove_urls(text):
    return re.sub(r'http\S+|www\.\S+', '', text)
```

* **remove\_urls**: Removes any URLs from the input text using regular expressions.

```python
def tokenize(text):
    return re.findall(r'\b\w+\b', text)
```

* **tokenize**: Tokenizes the input text into words (using regular expressions to match word boundaries).

```python
def replace_slang(tokens):
    return [slang_replacements.get(word, word) for word in tokens]
```

* **replace\_slang**: Replaces slang words with their formal counterparts based on the `slang_replacements` dictionary.

```python
def correct_spelling(tokens):
    return [spell.correction(w) if w.isalpha() and w not in slang_replacements else w for w in tokens]
```

* **correct\_spelling**: Corrects spelling errors using the `SpellChecker` library, unless the word is in the `slang_replacements` dictionary.

```python
def preprocess_text(text, keep_emotion=True):
    if not isinstance(text, str):
        return ""
    
    text = contractions.fix(text)
    text = remove_urls(text)
    text = emoji.demojize(text) if keep_emotion else emoji.replace_emoji(text, "")
    text = text.lower()
    text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)
    text = re.sub(r'[^a-z0-9\s!?<>:]', ' ', text)
    ...
```

* **preprocess\_text**: This function performs a series of preprocessing steps:

  * Expands contractions.
  * Removes URLs.
  * Converts emojis to their textual representation or removes them.
  * Makes the text lowercase.
  * Reduces repeated characters (e.g., "looooove" to "loove").
  * Removes non-alphanumeric characters.
  * Tokenizes the text, replaces slang, corrects spelling, and performs lemmatization.


### 6. **TF-IDF Vectorization and Cosine Similarity**

```python
class AgroChatbot:
    def __init__(self, faq_data):
        self.questions = [q for q, a in faq_data]
        self.answers = [a for q, a in faq_data]
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform([preprocess_text(q) for q in self.questions])
```

* **AgroChatbot Class**:

  * Initializes with `faq_data` and prepares the data by separating questions and answers.
  * **TfidfVectorizer**: Converts the questions into numerical feature vectors using TF-IDF.
  * The TF-IDF matrix is calculated for the preprocessed questions.

```python
    def get_response(self, user_input):
        user_clean = preprocess_text(user_input)
        user_vec = self.vectorizer.transform([user_clean])
        similarity = cosine_similarity(user_vec, self.tfidf_matrix)
        best_match = np.argmax(similarity)
        score = similarity[0][best_match]
        return self.answers[best_match] if score > 0.2 else "Sorry, I couldn't understand. Can you rephrase?"
```

* **get\_response**:

  * Preprocesses the user's input.
  * Transforms the input into a vector using the same vectorizer.
  * Calculates cosine similarity between the input and the preprocessed FAQ questions.
  * If the similarity score is greater than 0.2, returns the corresponding answer; otherwise, it prompts the user to rephrase.

### 7. **Main Execution Loop**

```python
if __name__ == "__main__":
    bot = AgroChatbot(faq_data)

    print("AGROBOT: Hello! Ask me anything about our AGRO FARMING..!!!")
    while True:
        print("    _Type::---'exit', 'quit','bye' for close.  ")
        user = input(" [ User ] : ")
        if user.lower() in ['exit', 'quit','bye']:
            break
        response = bot.get_response(user)
        print(f" [ AgroBot ] : {response}\n")
```

* Initializes the chatbot with the FAQ data.
* Greets the user and enters an infinite loop, awaiting user input.
* If the user types 'exit', 'quit', or 'bye', the program will break out of the loop and stop.
* The chatbot responds with the most relevant answer based on the similarity of the user's input to the FAQ questions.


## **Running the Chatbot**

To run the chatbot, simply execute the script. It will start the conversation and wait for user input. The bot will respond based on the most relevant FAQ answer.

---
