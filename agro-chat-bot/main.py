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
# Slang and emotion maps
slang_replacements = {
    'plz': 'please', 'thx': 'thanks', 'tnx': 'thanks', 'u': 'you', 'r': 'are',
    'ur': 'your', 'whr': 'where', 'ordr': 'order', 'btw': 'by the way', 'idk': 'i do not know',
    'imo': 'in my opinion', 'lol': 'laugh out loud', 'brb': 'be right back', 'omg': 'oh my god',
    'ttyl': 'talk to you later', 'pls': 'please', 'gr8': 'great', 'b4': 'before', 'l8r': 'later',
    '2': 'too', '4': 'for', 'cuz': 'because', 'bc': 'because', 'luv': 'love', 'msg': 'message',
    'txt': 'text', 'np': 'no problem', 'yw': 'you are welcome', 'afaik': 'as far as i know',
    'fyi': 'for your information', 'jk': 'just kidding', 'idc': 'i do not care', 'ty': 'thank you',
    'tyvm': 'thank you very much', 'k': 'okay', 'ok': 'okay'
}
emotion_map = {
    'angry': [':angry_face:', ':pouting_face:', ':rage:'],
    'happy': [':smile:', ':grinning_face:', ':laughing:', ':smiley:'],
    'sad': [':disappointed_face:', ':cry:', ':sob:'],
    'excited': [':star_struck:', ':tada:'],
    'confused': [':thinking_face:', ':confused_face:'],
    'frustrated': [':persevere:', ':weary:'],
    'surprised': [':open_mouth:', ':astonished:']
}
faq_data = (
    ("What are your opening hours?", "Our agro firm is open from 8:00 AM to 6:00 PM, Monday to Saturday."),
    ("Do you offer organic fertilizers?", "Yes, we have a wide range of organic fertilizers including compost, vermicompost, and bio-fertilizers."),
    ("What payment methods do you accept?", "We accept cash, bKash, Nagad, and credit/debit cards."),
    ("Do you provide delivery service for bulk orders?", "Yes, we offer free delivery for bulk orders above 10,000 BDT within Dhaka. Outside Dhaka, delivery charges apply."),
    ("Where is your main office located?", "Our main office is at 456 Agriculture Road, Farmgate, Dhaka."),
    ("Can I return a defective agricultural product?", "Yes, you can return defective products within 3 days of purchase with the original receipt."),
    ("Do you sell seeds for seasonal crops?", "We sell high-quality seeds for all seasonal crops including rice, wheat, vegetables, and fruits."),
    ("What types of pesticides do you offer?", "We offer both organic and chemical pesticides suitable for various crops and pests."),
    ("Do you provide agricultural consultancy services?", "Yes, we have expert agronomists who provide consultancy services at our office or farm visits (charges apply)."),
    ("How can I check product availability?", "You can call us at +880XXXXXXX or visit our website to check product availability."),
    ("Do you offer discounts for farmers' cooperatives?", "Yes, we provide special discounts for registered farmers' cooperatives and bulk buyers."),
    ("What safety measures do you have for COVID-19?", "We follow all government health guidelines including sanitization, masks, and social distancing at our outlets."),
    ("Who needs to complete a Census of Agriculture questionnaire?", 
     "Any person responsible for operating a farm or an agricultural operation should complete a Census of Agriculture questionnaire."),
    ("What is the definition of an agricultural operator?", 
     "An agricultural operator is a person responsible for the management and/or financial decisions in agricultural production. An operation can have multiple operators."),   
    ("How is an agricultural operation defined?", 
     "An agricultural operation is a farm or holding that produces agricultural products and reports revenues/expenses for tax purposes to the Canada Revenue Agency."),
    ("Are hobby farms included in the Census of Agriculture?", 
     "Yes, farms with very low revenues ('hobby farms') are included if they produce agricultural products and report revenues/expenses for tax purposes."),
    ("Why do operators of small operations have to complete the questionnaire?", 
     "Small operations contribute significantly to total agricultural inventories and must be counted for a complete statistical picture of Canada's farm sector."),
    ("What is the legal authority for the Census of Agriculture?", 
     "The mandate comes from the Constitution Act-1867 and Statistics Act-1970, requiring a census every 10 years (with an additional one every 5 years unless directed otherwise)."),
    ("Is it mandatory to answer and return the questionnaire?", 
     "Yes, under the Statistics Act, agricultural operators are required to complete a Census of Agriculture form."),
    ("Can a person be identified by the information they provide?", 
     "No. All published data are subject to confidentiality restrictions that prevent identification of individuals or operations."),
    ("Why does Statistics Canada conduct the Census of Agriculture?", 
     "To collect comprehensive data on Canada's agriculture industry including farm numbers, operators, land use, livestock, crops, expenses, receipts and equipment."),
    ("How does the Census of Agriculture benefit farm operators?", 
     "Operators use data for production/marketing decisions; groups use it for advocacy; governments for policy; businesses for investment; and media to inform the public."),
    ("Why doesn't the Census of Agriculture use sampling?", 
     "The Statistics Act requires counting every farm operation. Sampling wouldn't provide the complete picture needed for small-area data and survey benchmarking."),
    ("Why aren't there different questionnaires for different types of agricultural operations?", 
     "A single form ensures national consistency while tick boxes allow operators to answer only relevant questions, keeping costs down and minimizing burden."),
    ("How much does the Census of Agriculture cost?", 
     "The projected total cost for the 2021 Census of Agriculture over six years is $49.4 million."),   
    ("Why is the Census of Agriculture conducted in May, such a busy time for farmers?", 
     "Alignment with the Census of Population saves millions by sharing collection/processing resources. Population census timing aims to catch people when they're home."),
    ("What about my income tax return? The census seems to ask for similar information.", 
     "In 2021, respondents only provide total operating expenses and sales, with detailed expense questions removed to reduce burden."),
    ("How is response burden being reduced?", 
     "Options for online, mail or phone response; expanded collection window; toll-free helpline; and simplified questions all reduce burden."),
    ("How are Census of Agriculture data used?", 
     "Data are used by farmers, producer groups, governments, businesses, academics and media for decisions, policy, research and reporting."),
    ("When will the 2021 Census of Agriculture data be available?", 
     "First release was May 11, 2022. The Daily bulletin announces releases showing major trends and findings."),
    ("Why does it take a year to release results?", 
     "Collection, follow-up, quality checks, processing, validation and publication for all Canadian agricultural operations requires about one year."),
    ("For which geographical areas are data available?", 
     "Data are available for Canada, provinces/territories, and areas like counties, crop districts and rural municipalities."),
    ("Are cannabis farms included in the Census of Agriculture?", 
     "Cannabis operations were excluded from 2021 Census databases due to data quality issues, but separate statistics are available from Health Canada records."),
    ("How many agricultural operations were counted in the last Census of Agriculture?", 
     "The 2021 Census recorded 189,874 census farms, down from 193,492 in 2016."),
    ("Does the Census ask questions about farming's environmental impact?", 
     "Yes, questions cover soil conservation, pesticide/fertilizer use, land features, manure use, irrigation, tillage practices and crop residue management."),
    ("What's different about the 2021 Census from 2016?", 
     "2021 added questions on direct marketing, succession planning, greenhouse subcategories, renewable energy, and simplified response with Yes/No filters."),
    ("Where/how are questionnaires processed?", 
     "Online responses are captured automatically; paper forms go to a processing center for electronic capture with multiple quality checks."),
    ("What steps ensure all agricultural operations are counted?", 
     "Addresses from multiple sources, multiple response options, follow-up procedures, and processing safeguards identify missing or new operations.")
)

# Setup NLTK
NLTK_DATA_PATH = r'D:\ML\.venv\lib\nltk_data'
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_PATH)

for resource in ['stopwords', 'wordnet']:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, download_dir=NLTK_DATA_PATH)

spell = SpellChecker()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocessing Pipeline
def remove_urls(text):
    return re.sub(r'http\S+|www\.\S+', '', text)

def tokenize(text):
    return re.findall(r'\b\w+\b', text)

def replace_slang(tokens):
    return [slang_replacements.get(word, word) for word in tokens]

def correct_spelling(tokens):
    return [spell.correction(w) if w.isalpha() and w not in slang_replacements else w for w in tokens]

def preprocess_text(text, keep_emotion=True):
    if not isinstance(text, str):
        return ""
    
    text = contractions.fix(text)
    text = remove_urls(text)
    text = emoji.demojize(text) if keep_emotion else emoji.replace_emoji(text, "")
    text = text.lower()
    text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)
    text = re.sub(r'[^a-z0-9\s!?<>:]', ' ', text)

    tokens = tokenize(text)
    tokens = replace_slang(tokens)
    tokens = correct_spelling(tokens)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words or w in emotion_map]

    return ' '.join(sorted(set(tokens)))

# TF-IDF-based response engine
class AgroChatbot:
    def __init__(self, faq_data):
        self.questions = [q for q, a in faq_data]
        self.answers = [a for q, a in faq_data]
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform([preprocess_text(q) for q in self.questions])

    def get_response(self, user_input):
        user_clean = preprocess_text(user_input)
        user_vec = self.vectorizer.transform([user_clean])
        similarity = cosine_similarity(user_vec, self.tfidf_matrix)
        best_match = np.argmax(similarity)
        score = similarity[0][best_match]
        return self.answers[best_match] if score > 0.2 else "Sorry, I couldn't understand. Can you rephrase?"

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
