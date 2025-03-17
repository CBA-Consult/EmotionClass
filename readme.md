### --- Import the libraries and programs

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

### --- Function Definitions ---

#### Preproces Text
def preprocess_text(text):
    """Lowercase, tokenize, remove stop words and punctuation, and lemmatize."""
    try:
        # Explicitly load the Punkt sentence tokenizer
        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # Tokenize into sentences, *then* into words.  PASS THE TOKENIZER!
        sentences = sent_tokenizer.tokenize(text.lower())
        tokens = []
        for sent in sentences:
            words = word_tokenize(sent, language='english')  # Pass language here
            tokens.extend(words)

        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return tokens

    except LookupError as e:
        print(f"LookupError in preprocess_text: {e}")
        #Print helpful information:
        print(f"NLTK Data Path: {nltk.data.path}")
        import os
        print(f"NLTK_DATA environment variable: {os.environ.get('NLTK_DATA')}")
        print(f"Does the punkt file exist where expected? {os.path.exists(nltk.data.find('tokenizers/punkt/PY3/english.pickle'))}")
        return []  # Return an empty list on error
    except Exception as e:
        print(f"Unexpected error in preprocess_text: {e}")
        return []


#### Analyze Sentiment

def analyze_sentiment(text, lexicon_df):
    """Analyzes the sentiment of a text using the loaded lexicon data.

    Args:
        text (str): The text to analyze.
        lexicon_df (pd.DataFrame): The emotion lexicon DataFrame.

    Returns:
        dict: A dictionary of emotion scores for the text.
    """
    tokens = preprocess_text(text)
    emotion_scores = {
        'positive': 0,
        'negative': 0,
        'anger': 0,
        'anticipation': 0,
        'disgust': 0,
        'fear': 0,
        'joy': 0,
        'sadness': 0,
        'surprise': 0,
        'trust': 0
    }

    if lexicon_df.empty:
        print("Warning: Lexicon is empty. Returning zero scores.")
        return emotion_scores

    for word in tokens:
        # Case-insensitive check if the word exists in the lexicon
        if word.lower() in lexicon_df['word'].str.lower().values:
            word_row = lexicon_df[lexicon_df['word'].str.lower() == word.lower()].iloc[0]
            for emotion in emotion_scores.keys():
                emotion_scores[emotion] += int(word_row[emotion])

    return emotion_scores
### --- Main Program: Sentiment Analysis ---

### Load the *expanded* lexicon from the CSV file
expanded_lexicon_df = pd.read_csv("expanded_nrc_lexicon.csv")

### Example Usage (using the sample)
#### text1 = "This is a wonderfully happy and joyful day!"
#### text2 = "I am feeling sad, angry, and filled with fear."
#### text3 = "The movie was okay.  It wasn't amazing, but not terrible."
#### text4 = "The unexpected gift filled me with joy and surprise! I was so grateful."
#### text5 = "He felt abandoned and betrayed by his closest friends.  The injustice of it all made him furious."
#### text6 = "The looming deadline and the overwhelming workload created a sense of dread and anxiety."

#### scores1 = analyze_sentiment(text1, expanded_lexicon_df)
#### scores2 = analyze_sentiment(text2, expanded_lexicon_df)
#### scores3 = analyze_sentiment(text3, expanded_lexicon_df)
#### scores4 = analyze_sentiment(text4, expanded_lexicon_df)
#### scores5 = analyze_sentiment(text5, expanded_lexicon_df)
#### scores6 = analyze_sentiment(text6, expanded_lexicon_df)


print(f"Text 1 Scores: {scores1}")
print(f"Text 2 Scores: {scores2}")
print(f"Text 3 Scores: {scores3}")
print(f"Text 4 Scores: {scores4}")
print(f"Text 5 Scores: {scores5}")
print(f"Text 6 Scores: {scores6}")


### print(emotion_lexicon_df[emotion_lexicon_df['word'] == 'grief'])
### print(expanded_lexicon_df[expanded_lexicon_df['word'] == 'sorrow'])

#### Text 1 Scores: {'positive': 3, 'negative': 0, 'anger': 0, 'anticipation': 1, 'disgust': 0, 'fear': 0, 'joy': 3, 'sadness': 0, 'surprise': 1, 'trust': 2}
#### Text 2 Scores: {'positive': 1, 'negative': 4, 'anger': 4, 'anticipation': 1, 'disgust': 3, 'fear': 3, 'joy': 1, 'sadness': 2, 'surprise': 1, 'trust': 1}
#### Text 3 Scores: {'positive': 1, 'negative': 2, 'anger': 2, 'anticipation': 0, 'disgust': 2, 'fear': 2, 'joy': 1, 'sadness': 2, 'surprise': 0, 'trust': 1}
#### Text 4 Scores: {'positive': 5, 'negative': 1, 'anger': 0, 'anticipation': 2, 'disgust': 0, 'fear': 2, 'joy': 4, 'sadness': 0, 'surprise': 3, 'trust': 0}
#### Text 5 Scores: {'positive': 1, 'negative': 3, 'anger': 3, 'anticipation': 0, 'disgust': 1, 'fear': 1, 'joy': 1, 'sadness': 1, 'surprise': 0, 'trust': 1}
#### Text 6 Scores: {'positive': 2, 'negative': 2, 'anger': 1, 'anticipation': 2, 'disgust': 0, 'fear': 2, 'joy': 0, 'sadness': 1, 'surprise': 0, 'trust': 0}
       word  positive  negative  anger  anticipation  disgust  fear  joy  \
5627  grief         0         1      0             0        0     0    0   

      sadness  surprise  trust  
5627        1         0      0  
         word  positive  negative  anger  anticipation  disgust  fear  joy  \
11744  sorrow         0         1      0             0        0     1    0   

       sadness  surprise  trust  
11744        1         0      0
