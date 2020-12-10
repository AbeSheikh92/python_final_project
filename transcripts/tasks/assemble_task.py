import os
import pandas as pd
from functools import reduce
from luigi import ExternalTask, Task, Parameter
from luigi.util import inherits
from gensim.models import Word2Vec
from transcripts.word_embedding import construct_embedding
from transcripts.captions import compile_all_captions
from csci_utils.luigi.target import SuffixPreservingLocalTarget


class CreateAccumulatedModel(ExternalTask):  # pragma: no cover
    """
    Manages the compilation of all of the intended news organization's text
    files into one combined text file. This is done to allow for an ultra model
    to be trained using a substantially larger amount of caption data
    """

    # Parameter representing the local directory root
    CAPTIONS_ROOT = Parameter(default=os.path.abspath("data"))

    # Parameter representing directory path where model is to be saved
    # model_output_target_dir = Parameter(default="models/ALL_CAPTIONS")
    model_output_target_dir = Parameter(default=os.path.join("models", "ALL_CAPTIONS"))

    # Parameter representing path where model is to be saved
    trained_model_target_file = Parameter(default="trained_embedding.model")

    # Parameter representing which new organization's captions to compile
    news_organization = Parameter(default="cnn")

    def output(self):

        # Returns the target output file path
        paths_list = [
            str(self.CAPTIONS_ROOT),
            str(self.model_output_target_dir)
            + "_"
            + str(self.news_organization).upper(),
            str(self.trained_model_target_file),
        ]

        return SuffixPreservingLocalTarget(reduce(os.path.join, paths_list))

    def run(self):

        # Compiles the caption data
        compiled_captions_path = compile_all_captions(
            str(self.CAPTIONS_ROOT), str(self.news_organization)
        )

        # Reads in the caption data from the file
        with open(compiled_captions_path, mode="r") as input_target:
            all_captions_data = input_target.read()

        # Trains a new and shiny ultra model using the combined caption data
        ultra_model = construct_embedding(captions=all_captions_data, cbow=False)

        # Saves this new model to the output file path
        with self.output().open(mode="w") as output_target:
            ultra_model.save(output_target.name)


@inherits(CreateAccumulatedModel)
class QueryUltraModel(Task):  # pragma: no cover
    """
    Manages the querying of the ultra model
    """

    # Parameter representing term from which to search trained model for most similar words
    term_of_interest = Parameter(default="news")

    # Parameter representing path where model results are to be saved
    trained_model_results_target_file = Parameter(default="model_results.csv")

    def requires(self):
        return self.clone(CreateAccumulatedModel)

    def output(self):

        # Returns the target output file path
        paths_list = [
            str(self.CAPTIONS_ROOT),
            str(self.model_output_target_dir)
            + "_"
            + str(self.news_organization).upper(),
            "queries",
            str(self.term_of_interest)
            + "_"
            + str(self.trained_model_results_target_file),
        ]

        return SuffixPreservingLocalTarget(reduce(os.path.join, paths_list))

    def run(self):

        # Loads in the ultra model
        with self.input().open(mode="r") as input_target:
            loaded_ultra_model = Word2Vec.load(input_target.name)

        # Finds the words most similar to the term of interest
        similar_words = loaded_ultra_model.most_similar(
            str(self.term_of_interest), topn=30
        )

        # Convert data into a DataFrame
        similarity_df = pd.DataFrame(similar_words, columns=["word", "probability"])

        # Writes the DataFrame contents to a csv file
        with self.output().open(mode="w") as output_target:
            similarity_df.to_csv(output_target)

        # Displays the results
        self.print_results()

    def print_results(self):
        print(pd.read_csv(self.output().path))
