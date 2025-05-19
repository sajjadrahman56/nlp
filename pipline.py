import re
import emoji
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker 

# Setup NLTK data path if needed
NLTK_DATA_PATH = r'D:\ML\.venv\lib\nltk_data'
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_PATH)

# Download required NLTK data
for resource in ['stopwords', 'wordnet']:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, download_dir=NLTK_DATA_PATH)

spell = SpellChecker()  # Initialize spellchecker

def remove_urls(text):
    return re.sub(r'http\S+|www\.\S+', '', text)

def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text)

slang_replacements = {
    'plz': 'please', 'thx': 'thanks', 'tnk': 'thanks', 'tnx': 'thanks', 'u': 'you', 'r': 'are',
    'ur': 'your', 'whr': 'where', 'ordr': 'order', 'btw': 'by the way', 'idk': 'i do not know',
    'imo': 'in my opinion', 'lol': 'laugh out loud', 'brb': 'be right back', 'omg': 'oh my god',
    'ttyl': 'talk to you later', 'pls': 'please', 'gr8': 'great', 'b4': 'before', 'l8r': 'later',
    '2': 'too', '4': 'for', 'cuz': 'because', 'bc': 'because', 'luv': 'love', 'msg': 'message',
    'txt': 'text', 'np': 'no problem', 'yw': 'you are welcome', 'afaik': 'as far as i know',
    'fyi': 'for your information', 'jk': 'just kidding', 'idc': 'i do not care', 'ty': 'thank you',
    'tyvm': 'thank you very much', 'k': 'okay', 'ok': 'okay',
}
emotion_map = {
        'angry': ['angry', ':angry_face:', ':pouting_face:', ':face_with_symbols_over_mouth:', ':rage:'],
        'happy': ['happy', ':smile:', ':grinning_face:', ':laughing:', ':blush:', ':smiley:'],
        'sad': ['sad', ':disappointed_face:', ':cry:', ':sob:', ':worried:'],
        'frustrated': ['frustrated', ':persevere:', ':confounded_face:', ':weary:', ':triumph:'],
        'confused': ['confused', ':thinking_face:', ':confused_face:', ':hushed:'],
        'excited': ['excited', ':star_struck:', ':partying_face:', ':tada:'],
        'surprised': ['surprised', ':open_mouth:', ':astonished:'],
    }
def replace_slang(tokens):
    return [slang_replacements.get(token, token) for token in tokens]

def correct_spelling(words):
    corrected = []
    for word in words:
        # Only correct if word is alphabetic and not slang or special tokens
        if word.isalpha() and word not in slang_replacements:
            corrected_word = spell.correction(word)
            corrected.append(corrected_word)
        else:
            corrected.append(word)
    return corrected

def detect_shouting(text):
    shouting_words = re.findall(r'\b[A-Z]{2,}\b', text)
    return len(shouting_words) > 0

def detect_emotions(original_text, demojized_text):
    detected = set()
    orig_lower = original_text.lower()
    demo_lower = demojized_text.lower()
    for emotion, keywords in emotion_map.items():
        for kw in keywords:
            if kw in orig_lower or kw in demo_lower:
                detected.add(emotion)
                break
    return detected

def preprocess_text(text, keep_emotion=True):
    if not text or not isinstance(text, str):
        return ""

    original_text = text
    shouting = detect_shouting(original_text)

    # Expand contractions
    text = contractions.fix(text)

    # Add shouting token before lowering
    if shouting:
        text += " <shouting>"

    text = text.lower()
    text = remove_urls(text)

    # Emoji handling
    if keep_emotion:
        text = emoji.demojize(text)
        # Collapse repeated emoji tokens like ':angry_face: :angry_face:' → ':angry_face:'
        text = re.sub(r'(:[a-z_]+:)( \1)+', r'\1', text)
    else:
        text = emoji.replace_emoji(text, "")

    # Limit repeated letters to max 3
    text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)

    # Clean special chars but keep !?<>: (emoji tokens)
    if keep_emotion:
        text = re.sub(r'[^a-z0-9\s!?<>:]', ' ', text)
    else:
        text = re.sub(r'[^a-z0-9\s]', ' ', text)

    words = simple_tokenize(text)
    words = replace_slang(words)
    words = correct_spelling(words) 

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Remove stopwords but keep important/emotion words
    filler_words = set(stopwords.words('english'))
    important_words = {'angry', 'happy', 'sad', 'urgent', 'late', 'not', 'no', 'never',
                       'shouting', 'frustrated', 'confused', 'excited', 'surprised'}
    words = [w for w in words if w not in filler_words or w in important_words]

    if keep_emotion:
        emotions = detect_emotions(original_text, text)
        words.extend(emotions)

    # ✅ Remove duplicate words before final output
    seen = set()
    words = [w for w in words if not (w in seen or seen.add(w))]

    clean_text = ' '.join(words)
    clean_text = ' '.join(clean_text.split())


    return clean_text

if __name__ == "__main__":
    test_messages = [
        "Hey, can u plz tell me where's my order?? Visit https://example.com for details.",
        "I didn't receive my parcel yet!!!! 😡 Check www.tracking.com",
        "Whr's my ordr 😡😡",
        "delivery late af... I want refund now",
        "OMG!!! THIS IS RIDICULOUS!!! 😠😠😠",
        "I'm so happy with my order! 😊😊",
        "Feeling sad and desappointed... 😞",
        "What??? WHY IS MY ORDER LATE???",
        "This is so frustrating!!! 😤😤",
        "LOL that was funny tooooooooooooooo 😂😂",
    ]

    print("CLEANING CUSTOMER MESSAGES:")
    print("=" * 50)
    for msg in test_messages:
        print(f"Original: {msg}")
        print(f"Cleaned: {preprocess_text(msg)}")
        print("-^-" * 10)
