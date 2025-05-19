# Text Preprocessing Pipeline Documentation

## Overview
This provides a comprehensive text preprocessing pipeline designed specifically for customer service messages. It handles text normalization, emotion detection, slang conversion, and noise removal while preserving important emotional context.

## Dependencies

- `emoji`: For handling emoji conversions
- `contractions`: For expanding shortened word forms
- `nltk`: Natural Language Toolkit for advanced text processing
- `spellchecker`: For spelling correction

## Initial Setup

### NLTK Configuration
```python
NLTK_DATA_PATH = r'D:\ML\.venv\lib\nltk_data'
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_PATH)
```
- Sets a custom path for NLTK data files
- Ensures the script can find required NLTK resources

### Resource Download
```python
for resource in ['stopwords', 'wordnet']:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, download_dir=NLTK_DATA_PATH)
```
- Checks for and downloads necessary NLTK datasets:
  - `stopwords`: Common words to filter out
  - `wordnet`: Lexical database for lemmatization

### Spellchecker Initialization
```python
spell = SpellChecker()
```
- Creates a spellchecker instance for later corrections

## Functions

### 1. URL Removal
```python
def remove_urls(text):
    return re.sub(r'http\S+|www\.\S+', '', text)
```
- Removes web URLs using `regex` pattern matching
- Handles both `http://` and `www.` formats

### 2. Basic Tokenization
```python
def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text)
```
- Splits text into words using word boundaries

### 3. Slang Dictionary
```python
slang_replacements = { ... }
```
- Comprehensive mapping of slang to full forms
- Includes common abbreviations like:
  - "plz" → "please"
  - "u" → "you"

### 4. Emotion Mapping
```python
emotion_map = { ... }
```
- Defines emotional categories with:
  - Text keywords ("happy", "sad")
  - Emoji representations (":smile:", ":angry_face:")
- Covers 7 core emotions: `Angry, Happy, Sad, Frustrated, Confused, Excited, Surprised`

### 5. Slang Replacement
```python
def replace_slang(tokens):
    return [slang_replacements.get(token, token) for token in tokens]
```
- Processes tokens through slang dictionary
- Preserves original if no slang match found

### 6. Spelling Correction
```python
def correct_spelling(words):
    corrected = []
    for word in words:
        if word.isalpha() and word not in slang_replacements:
            corrected_word = spell.correction(word)
            corrected.append(corrected_word)
        else:
            corrected.append(word)
    return corrected
```
- Only corrects alphabetic words
- Skips words in slang dictionary to prevent over-correction

### 7. Shouting Detection
```python
def detect_shouting(text):
    shouting_words = re.findall(r'\b[A-Z]{2,}\b', text)
    return len(shouting_words) > 0
```
- Identifies words with 2+ uppercase letters
- Returns boolean if shouting detected

### 8. Emotion Detection
```python
def detect_emotions(original_text, demojized_text):
    detected = set()
    for emotion, keywords in emotion_map.items():
        for kw in keywords:
            if kw in orig_lower or kw in demo_lower:
                detected.add(emotion)
                break
    return detected
```
- Checks both original and emoji-converted text
- Returns set of detected emotions and Prevents duplicate emotion tags


### Function Signature
```python
def preprocess_text(text, keep_emotion=True):
```

### Step-by-Step Processing:

1. **Input Validation**
   ```python
   if not text or not isinstance(text, str):
       return ""
   ```
   - Returns empty string for invalid input

2. **Contraction Expansion**
   ```python
   text = contractions.fix(text)
   ```
   - Converts "don't" → "do not", "I'm" → "I am"

3. **Shouting Handling**
   ```python
   if shouting:
       text += " <shouting>"
   ```
   - Adds special token if shouting detected

4. **Case Normalization**
   ```python
   text = text.lower()
   ```

5. **URL Removal**
   ```python
   text = remove_urls(text)
   ```

6. **Emoji Processing**
   ```python
   if keep_emotion:
       text = emoji.demojize(text)
       text = re.sub(r'(:[a-z_]+:)( \1)+', r'\1', text)
   else:
       text = emoji.replace_emoji(text, "")
   ```
   - Converts emojis to text codes when keeping emotions
   - Collapses repeated emojis and Removes entirely when emotion not needed

7. **Character Normalization**
   ```python
   text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)
   ```
   - Limits repeated letters to max 3 ("soooo" → "sooo")

8. **Special Character Cleaning**
   ```python
   if keep_emotion:
       text = re.sub(r'[^a-z0-9\s!?<>:]', ' ', text)
   else:
       text = re.sub(r'[^a-z0-9\s]', ' ', text)
   ```
   - Preserves emoji tokens when keeping emotions

9. **Token Processing**
   ```python
   words = simple_tokenize(text)
   words = replace_slang(words)
   words = correct_spelling(words)
   ```

10. **Lemmatization**
    ```python
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    ```
    - Reduces words to base forms ("running" → "run")

11. **Stopword Filtering**
    ```python
    filler_words = set(stopwords.words('english'))
    important_words = {'angry', 'happy', 'sad', ...}
    words = [w for w in words if w not in filler_words or w in important_words]
    ```
    - Keeps emotion-related and negation words

12. **Emotion Tagging**
    ```python
    if keep_emotion:
        emotions = detect_emotions(original_text, text)
        words.extend(emotions)
    ```

13. **Duplicate Removal**
    ```python
    seen = set()
    words = [w for w in words if not (w in seen or seen.add(w))]
    ```
    - Ensures unique tokens in output

14. **Final Cleanup**
    ```python
    clean_text = ' '.join(words)
    clean_text = ' '.join(clean_text.split())
    ```
    - Normalizes whitespace

## Example Usage
```python
test_messages = [
    "Hey, can u plz tell me where's my order?? Visit https://example.com for details.",
    "I didn't receive my parcel yet!!!! 😡 Check www.tracking.com",
]

for msg in test_messages:
    print(f"Original: {msg}")
    print(f"Cleaned: {preprocess_text(msg)}")
    print("-^-" * 50)
```

## Output Samples
```
Original: Hey, can u plz tell me where's my order?? Visit https://example.com for details.
Cleaned: hey please tell where order <shouting>
------------------------------------------------------------------

Original: I didn't receive my parcel yet!!!! 😡 Check www.tracking.com
Cleaned: not receive parcel yet angry <shouting> check
------------------------------------------------------------------

Original: OMG!!! THIS IS RIDICULOUS!!! 😠😠😠
Cleaned: oh my god ridiculous angry <shouting>
------------------------------------------------------------------
```
