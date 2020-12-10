import os
import pandas as pd
from functools import reduce
from luigi import Task, Parameter
from luigi.util import inherits
from gensim.models import Word2Vec
from csci_utils.luigi.target import SuffixPreservingLocalTarget
from .caption_tasks import ProcessCaptionData
from ..word_embedding import construct_embedding


@inherits(ProcessCaptionData)
class FeedToModel(Task):  # pragma: no cover
    """
    Manages the feeding of the accumulated caption data to
    the Word2Vec word embedding model and thus the
    construction of the model
    """

    # Parameter representing the local file root
    MODEL_ROOT = Parameter(default=os.path.abspath(os.path.join("data", "models")))

    # Parameter representing the captions target file
    video_captions_target_file = Parameter(default="cnn_captions.txt")

    # Parameter representing the target file to save the model to
    trained_model_target_file = Parameter(default="trained_embedding.model")

    def requires(self):
        return self.clone(ProcessCaptionData)

    def output(self):

        # Returns the output target file path. The path incorporates
        # the captions target file path for a partial salting.

        """
        paths_list = [
            str(self.MODEL_ROOT),
            os.path.splitext(str(self.video_captions_target_file))[0],
            str(self.trained_model_target_file),
        ]

        return SuffixPreservingLocalTarget(reduce(os.path.join, paths_list))
        """

        return SuffixPreservingLocalTarget(
            os.path.join(
                (
                    os.path.join(
                        "%s" % self.MODEL_ROOT,
                        "%s"
                        % os.path.splitext(str(self.video_captions_target_file))[0],
                    )
                ),
                "%s" % self.trained_model_target_file,
            )
        )

    def run(self):

        # Reads in the caption data
        with self.input().open(mode="r") as input_target:
            captions_data = input_target.read()

        # Feeds data to and creates model
        trained_model = construct_embedding(captions=captions_data, cbow=True)

        # Saves model to target file
        with self.output().open(mode="w") as output_target:
            trained_model.save(output_target.name)


@inherits(FeedToModel)
class AnalyzeModelResults(Task):  # pragma: no cover
    """
    Manages the analysis of the model results:
    1. Term of interest is fed to trained model
    2. Model returns top N words that are most similar

    This similarity decision is made based on the cosine
    similarity of the words' vector representations
    """

    # Parameter representing the local data root
    RESULTS_ROOT = Parameter(default=os.path.abspath(os.path.join("data", "models")))

    # Parameter representing the captions target file
    video_captions_target_file = Parameter(default="cnn_captions.txt")

    # Parameter representing the model results target file
    model_results_target_file = Parameter(default="model_results.csv")

    # Parameter representing the term for which to retrieve
    # the top N similar words
    search_term = Parameter(default="news")

    def requires(self):
        return self.clone(FeedToModel)

    def output(self):

        """
        paths_list = [
            str(self.RESULTS_ROOT),
            os.path.splitext(str(self.video_captions_target_file))[0],
            str(self.model_results_target_file),
        ]

        return SuffixPreservingLocalTarget(reduce(os.path.join(paths_list)))
        """

        # Returns the target file path for the model results
        return SuffixPreservingLocalTarget(
            os.path.join(
                (
                    os.path.join(
                        "%s" % self.RESULTS_ROOT,
                        "%s"
                        % os.path.splitext(str(self.video_captions_target_file))[0],
                    )
                ),
                "%s" % self.model_results_target_file,
            )
        )

    def run(self):

        # Loads the saved model
        with self.input().open(mode="r") as input_target:
            loaded_model = Word2Vec.load(input_target.name)

        # Determines the words that are most similar to the term of interest
        most_similar_words = loaded_model.similar_by_word(
            str(self.search_term), topn=30
        )

        # Stores results to DataFrame
        words_df = pd.DataFrame(most_similar_words, columns=["word", "probability"])

        # Writes results to target file
        with self.output().open(mode="w") as output_target:
            words_df.to_csv(output_target)

        # Displays results
        self.print_results()

    def print_results(self):
        print(pd.read_csv(self.output().path))
