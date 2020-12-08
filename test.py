import os
from unittest import TestCase
from gensim.models import Word2Vec
from transcripts.captions import get_video_ids
from transcripts.word_embedding import construct_embedding
from transcripts.captions import compile_all_captions
from transcripts.data_cleaning import (
    remove_punctuation,
    remove_stop_words,
    gen_sample_data,
    is_english,
    check_gibberish,
)


"""
NOTE: I refrained from testing the luigi tasks directly for the following three reasons:
----------------------------------------------------------------------------------

1. The grand majority of their logic is comprised of functions in other files which are already tested below.

2. They make calls to a third-party API with no known 'mock' type testing component
   so they would excessively rely on remote data

3. Professor Gorlin has repeatedly mentioned that testing the internals of
   luigi task logic is essentially meaningless.
"""


class DataCleaningTests(TestCase):
    """
    Ensures the data cleaning preprocessing steps are functioning correctly
    """

    def test_remove_punctuation(self):
        """
        Tests the removal of punctuation from 2D tokenized lists
        """

        # Represents a 2D tokenized list WITH punctuation
        test_list_with_punctuation = [
            ["this", "is", "a", "tokenized", "sentence", "."],
            ["now", "this", "is", "another", "sentence", "!"],
            ["these", ",", "things", "!", "are", "punctuations"],
        ]

        # Represents a 2D tokenized list WITHOUT punctuation
        test_list_without_punctuation = [
            ["this", "is", "a", "tokenized", "sentence"],
            ["now", "this", "is", "another", "sentence"],
            ["these", "things", "are", "punctuations"],
        ]

        # Ensures the punctuations are properly removed from the unfiltered list
        self.assertEqual(
            remove_punctuation(test_list_with_punctuation),
            test_list_without_punctuation,
        )

    def test_remove_stop_words(self):
        """
        Tests the removal of stop words from 2D tokenized lists
        """

        # Represents a 2D tokenized list WITH stop words
        test_list_with_stop_words = [
            ["this", "is", "a", "sentence"],
            ["now", "this", "is", "another", "sentence"],
            ["third"],
        ]

        # Represents a 2D tokenized list WITHOUT stop words
        test_list_without_stop_words = [
            ["sentence"],
            ["another", "sentence"],
            ["third"],
        ]

        # Ensures the stop words are properly removed from the unfiltered list
        self.assertEqual(
            remove_stop_words(test_list_with_stop_words), test_list_without_stop_words
        )

    def test_gen_sample_data(self):
        """
        Test the proper creation of sample data
        """

        # Represents a list of caption data (stored as a list of dictionaries) from YouTube
        test_whole_data = [
            {"text": "this represents youtube captions"},
            {"text": "oh look, more youtube captions"},
            {"text": "I have spent so many hours on this project"},
            {"text": "i can feel the sanity fading"},
        ]

        # Represents the list of extracted captions from the above sample of caption data
        test_sample_data = [
            "this represents youtube captions",
            "oh look, more youtube captions",
            "I have spent so many hours on this project",
            "i can feel the sanity fading",
        ]

        # Ensures proper extraction of YouTube caption text data
        self.assertEqual(gen_sample_data(test_whole_data), test_sample_data)

    def test_is_english(self):
        """
        Tests whether the extracted YouTube caption text is quality text by checking whether a
        small sample of consecutive words are recognizably english words
        """

        # Represents a sample of YouTube caption text data
        test_words = ["this", "is", "definitely", "ald;kjs", "wl;33lj"]

        # Represents the words from the above list that should survive the non-english-word purge
        words_that_survived = ["this", "is", "definitely"]

        checked_words = [word for word in test_words if is_english(word.lower())]

        # Ensures the surviving list is comprised of the correct words
        self.assertListEqual(checked_words, words_that_survived)

    def test_check_gibberish(self):
        """
        Tests whether the extracted YouTube caption text is quality text by
        checking whether at least 80% of a small sample of consecutive words
        are recognizably english words (builds off of the 'is_english' function)

        This is a crucial step since apparently a not-so-insignificant percentage
        of YouTube captions are literally gibberish or not properly translated to
        English even when requested to be. This subset of non-english words will
        adversely affect the performance of the word embedding model since the
        model would literally be factoring in nonsense.

        Thus, part of the data pre-processing procedure involves checking a small
        sample of each YouTube video's caption data for 'gibberish'. Video's that
        do not survive this purge do NOT have their captions included in the model
        training.
        """

        # Represents a list of YouTube caption data that should survive the gibberish-purge
        # since it is comprised of at least 80% of recognizably english words
        sensible_list = [
            "this",
            "is",
            "definitely",
            "a",
            "sensible",
            "list",
            "which",
            "makes",
            "sense",
        ]

        # Represents a list of YouTube caption data that should NOT survive the gibberish-purge
        gibberish_list = [
            "this",
            "is",
            "definitely",
            "slkjwer",
            ">>wlke",
            "32234",
            "NOT",
            "2wkjl",
        ]

        # Ensures the sensible list is NOT categorized as 'gibberish'
        self.assertFalse(check_gibberish(sensible_list))

        # Ensures the gibberish list IS categorized as 'gibberish'
        self.assertTrue(check_gibberish(gibberish_list))


"""
class CaptionsTests(TestCase):
    # Tests the proper accessing of YouTube video Id elements via
    # the caption API

    def setUp(self) -> None:
        # Executes prior to every test ensuring test output files do not already exist

        target_path = os.path.join(os.path.abspath('data'), 'test.txt')
        if os.path.exists(target_path):
            os.remove(target_path)

    def tearDown(self) -> None:
        # Executes after every test ensuring test output files do not still exist

        self.setUp()

    def test_get_video_ids(self):
        # This is the only test that makes a quick YouTube call (DESIGNED FOR CHROME!!!)
        
        # Calls the caption API requesting a subset of YouTube video Ids
        # related to the 'anderson cooper' YouTube search
        video_ids_set = get_video_ids("anderson cooper", 0, 0, "by cnn")

        # Ensures the returned unique set of video Ids is not empty
        self.assertTrue(len(video_ids_set) > 0)
    
    def test_compile_all_captions(self):
        # Ensures files are properly concatenated when 'compile_all_functions' is called

        # Represents the root directory of files to concatenate
        test_root = os.path.abspath("data")

        # Represents one of the filters by which to search through files.
        # Files without 'cnn' in their names will be filtered out.
        test_news_organization = "cnn"

        # 'Manually' computes the total sizes of the intended files
        total_file_length = sum(
            [
                os.path.getsize(os.path.join(test_root, file))
                for file in os.listdir(test_root)
                if "captions" in file and test_news_organization in file
            ]
        )

        # Performs compilation
        target_path = compile_all_captions(
            test_root, test_news_organization, testing=True
        )

        # Ensures compilation file size is correct. This is of course
        # an estimate of how accurate the function being tested is.
        self.assertEqual(os.path.getsize(target_path), total_file_length)
"""


class WordEmbeddingTests(TestCase):
    """
    Tests the proper construction of a Word2Vec word embedding
    model with a simple test string of fake caption data
    """

    def test_construct_embedding(self):

        # Constructs the model
        word_vec_model = construct_embedding(
            "this is a string of characters that is called a sentence"
        )

        # Ensures the returned object is indeed a Word2Vec object
        self.assertIsInstance(word_vec_model, Word2Vec)
