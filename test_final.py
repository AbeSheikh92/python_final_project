import os
from unittest import TestCase
from tempfile import TemporaryDirectory
from gensim.models import Word2Vec
from atomicwrites import atomic_write
from transcripts.word_embedding import construct_embedding
from transcripts.captions import compile_all_captions, write_to_file
from transcripts.data_cleaning import (
    remove_punctuation,
    remove_stop_words,
    gen_sample_data,
    is_english,
    check_gibberish,
)
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')


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


class CaptionsTests(TestCase):
    """
    Tests the proper accessing of YouTube video Id elements via
    the caption API
    """

    def ensure_files_removed(self, file_path):
        """
        Ensures certain files created during testing are completely removed
        :param file_path: file path representing test output target
        """

        if os.path.exists(file_path):
            os.remove(file_path)

    def test_write_to_file(self):
        """
        Ensures the successful retrieval of YouTube
        caption data via the corresponding API
        """

        # Test YouTube video id
        test_video_id = ['nnPgZOBsRkM']

        # Returned caption data from above video
        returned_captions = write_to_file(test_video_id)

        # Ensures the returned information is a string
        self.assertTrue(isinstance(returned_captions, str))

        # Ensures the returned string is larger than 100.
        # This just serves as an estimate that the string is
        # most likely caption data and not some other random string.
        self.assertTrue(len(returned_captions) > 100)
    
    def test_compile_all_captions(self):
        """
        Ensures the functionality to compile the filtered text file results functions correctly
        """

        # Creates three temporary files within a temporary directory
        with TemporaryDirectory() as tmp:

            # The below three files represent three test text files
            temp_file_path_1 = os.path.join(tmp, 'cnn_test_1_captions.txt')
            temp_file_path_2 = os.path.join(tmp, 'cnn_test_2_captions.txt')
            temp_file_path_3 = os.path.join(tmp, 'cnn_test_3_captions.txt')

            # The below three strings represent three test text file contents
            temp_file_1_contents = 'cnn captions data part 1'
            temp_file_2_contents = 'cnn captions data part 2'
            temp_file_3_contents = 'cnn captions data part 3'

            # Writes temp_file_path_1 contents into file
            with atomic_write(temp_file_path_1, mode='w') as output_file_1:
                output_file_1.write(temp_file_1_contents)

            # Writes temp_file_path_2 contents into file
            with atomic_write(temp_file_path_2, mode='w') as output_file_2:
                output_file_2.write(temp_file_2_contents)

            # Writes temp_file_path_3 contents into file
            with atomic_write(temp_file_path_3, mode='w') as output_file_3:
                output_file_3.write(temp_file_3_contents)

            # Represents the directory root where these text files live
            test_root = tmp

            # Represent the new organization for which to filter text files by
            test_new_organization = 'cnn'

            # The returned file path that stores the concatenated results of the above test text files
            target_path = compile_all_captions(test_root, test_new_organization)

            # The combined file size of the three test text files as calculated by summing their lengths
            total_file_length = sum([len(temp_file_1_contents), len(temp_file_2_contents), len(temp_file_3_contents)])

            # The combined file size of the three test text file after
            # they have concatenated by the 'compile_all_captions' function
            target_file_size = os.path.getsize(target_path)

            # Ensures both lengths are equal.
            # This serves as one estimate of the accuracy of the 'compile_all_captions' function.
            # One reason I don't check to see if the actual written contents equals the
            # concatenation of the above three test text files is because for some reason
            # the 'compile_all_captions' function does not concatenate in the order
            # the files are written to / searched for within the function itself. I have
            # manually inspected that it indeed is concatenating them correctly.
            # Properly automating this type of inspection is another improvement that can be
            # made to this project.
            self.assertEqual(total_file_length, target_file_size)

        # Added redundancy on top of using TemporaryDirectory
        # to ensures the temporary files are indeed removed
        self.ensure_files_removed(temp_file_path_1)
        self.ensure_files_removed(temp_file_path_2)
        self.ensure_files_removed(temp_file_path_3)


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
