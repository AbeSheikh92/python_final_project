from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
import warnings
from transcripts.data_cleaning import remove_stop_words, remove_punctuation

# Filters a particular warning from being printed to the console
warnings.filterwarnings(action="ignore")


def construct_embedding(captions, cbow=True):
    """
    Utilizes 'transcripts.data_cleaning.py' to preprocess the YouTube
    caption data which this method then uses to construct a trained
    Word2Vec model.

    :param captions: str representing the entire concatenated
                     captions from all the videos retrieved from the search
    :param cbow: boolean True / False representing whether the model with utilized
                 the CBOW or Skip-Gram model architecture
    :return: Trained Word2Vec model
    """

    # List of characters to filter out of the caption string
    chars_to_remove = ["\n", ">", "--"]

    for char in chars_to_remove:
        captions = captions.replace(char, " ")

    # Perform some necessary tokenization
    data = [word_tokenize(word.lower()) for word in sent_tokenize(captions)]

    # Filter out punctuation from the data
    data_minus_punctuation = remove_punctuation(data)

    # Filter out stop words from the data
    data_minus_stop_words = remove_stop_words(data_minus_punctuation)

    # Filter out any surviving single character list elements
    fully_formatted_data = [
        [word for word in item if len(word) > 1] for item in data_minus_stop_words
    ]

    # Train the Word2Vec model using the specified architecture
    if cbow:
        print("UTILIZING CBOW ARCHITECTURE")
        model = Word2Vec(fully_formatted_data, min_count=1, size=100, window=5)
    else:
        print("UTILIZING SKIPGRAM ARCHITECTURE")
        model = Word2Vec(fully_formatted_data, min_count=1, size=100, window=5, sg=1)

    return model
