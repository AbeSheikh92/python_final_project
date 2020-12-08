from functools import wraps
from nltk.corpus import words
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def remove_punctuation(list_to_filter):
    """
    Takes a 2D tokenized list and removes any elements that are punctuations

    :param list_to_filter: 2D tokenized list
    :return: 2D tokenized list (filtered for punctuations)
    """

    data_minus_punctuation = [
        [word.lower() for word in item if word.isalpha()] for item in list_to_filter
    ]
    return data_minus_punctuation


def remove_stop_words(list_to_filter):
    """
    Takes a 2D tokenized list and removes any elements that are stop words
    as recognized by the nltk.corpus library

    :param list_to_filter: 2D tokenized list
    :return: 2D tokenized list (filtered for stop words)
    """

    stop_words = set(stopwords.words("english"))
    data_minus_stop_words = [
        [word for word in item if not word in stop_words] for item in list_to_filter
    ]
    return data_minus_stop_words


def gen_sample_data(sample_list):
    """
    Creates a subset of the captions data to be tested in order
    to determine whether the text is recognizably english or not

    :param sample_list: list of YouTube caption data (stored as list of dictionaries)
    :return: a subset list of the passed in argument
    """

    sample_data = [sample_list[i]["text"] for i in range(len(sample_list)) if i < 10]
    return sample_data


def is_english(word_to_check):
    """
    Determines whether a word is english or not as
    recognized by the nltk.corpus library

    :param word_to_check: str word to check
    :return: boolean True / False
    """

    if word_to_check in words.words():
        return True
    else:
        return False


def replace_chars(func, chars_to_remove=("'", ">", "-")):
    """
    Decorator that removes certain characters from the passed in data

    :param func: the function that is to be decorated
    :param chars_to_remove: a set of characters that is to be filtered out of the incoming data
    :return: a list of filtered text
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        # Removes the specified characters from each element of the list
        new_args = list(
            map(
                lambda x: "".join([char for char in x if char not in chars_to_remove]),
                args[0],
            )
        )

        # Reassembles the data into a list with each element as a single word
        new_args = word_tokenize(" ".join(new_args))
        return func(new_args, **kwargs)

    return wrapper


@replace_chars
def check_gibberish(data_to_check):
    """
    Determines whether or not each element of the passed in list is gibberish or not
    by strategy of sending the element to the 'is_english' function and then
    computing the fraction of element that were determined to be english.

    If this fraction is at least 80%, the list of caption data is considered
    to be sensible english. If not, it is considered to be gibberish and the
    caption data for the video is discarded.

    The 'replace_chars' decorator takes care of processing the words such that
    they are more recognizable to the 'nltk.corpus.words' library used by the 'is_english'
    function, thereby allowing 'check_gibberish' to offload that component of the preprocessing
    to another part of the code.

    :param data_to_check: list of strings representing YouTube captions data
    :return: boolean True / False depending on how much of the list is considered to be english
    """

    check_list = [
        is_english(word.lower()) * 1
        for word in data_to_check
        if len(word) > 1 and word[-1] != "s"
    ]

    # try different values for the threshold
    return sum(check_list) / float(len(check_list)) < 0.80
