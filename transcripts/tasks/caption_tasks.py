import os
from luigi import ExternalTask, Task, Parameter, IntParameter
from luigi.util import inherits
from csci_utils.luigi.target import SuffixPreservingLocalTarget
from ..captions import get_video_ids, write_to_file


class GetYoutubeVideoIds(ExternalTask):  # pragma: no cover
    """
    Manages the retrieval of YouTube video Ids related
    to the search query
    """

    # Parameter representing the local directory root
    LOCAL_ROOT = Parameter(default=os.path.abspath("data"))

    # Parameter representing the file path to stop the video Ids
    video_ids_target_file = Parameter(default="cnn_video_ids.txt")

    # Parameter representing the YouTube search query
    query = Parameter(default="cnn immigration")

    # Parameter representing the number of pixels to
    # scroll the loaded YouTube page downwards
    scroll_volume = IntParameter(default=0)

    # Parameter representing the number of times to scroll
    # the screen downwards (each scroll scrolls downwards
    # by the 'scroll_volume' pixel amount)
    num_cycles = IntParameter(default=0)

    # Parameter representing the YouTube channel uploaded
    channel_author = Parameter(default="by cnn")

    def output(self):

        # Returns the target file path for the video Ids
        return SuffixPreservingLocalTarget(
            os.path.join("%s" % self.LOCAL_ROOT, "%s" % self.video_ids_target_file)
        )

    def run(self):

        # Retrieves video Ids and writes them to the target file path
        video_list = get_video_ids(
            self.query, self.scroll_volume, self.num_cycles, self.channel_author
        )

        with self.output().open(mode="w") as output_target:
            for video in video_list:
                output_target.write("%s\n" % video)


@inherits(GetYoutubeVideoIds)
class ProcessCaptionData(Task):  # pragma: no cover
    """
    Manages the processing of YouTube captions which involves (for each video Id):
    1. Retrieving the captions

    2. Removing certain characters that will confuse the interpretation of words as actual words

    3. Testing a sample of consecutive caption text for at least 80% of recognizably english words.
       This is done in order to filter out videos who's captions data is either poorly
       created or unavailable in english. This step is carried out in order to prevent
       essentially nonsensical words diluting the Word2Vec model. The threshold of 80%
       should be high enough to filter out substantially gibberish captions while
       allowing for a slight amount of caption errors.
    """

    # Parameter representing the local directory root
    CAPTIONS_ROOT = Parameter(default=os.path.abspath("data"))

    # Parameter representing the target file path for the caption data
    video_captions_target_file = Parameter(default="cnn_captions.txt")

    def requires(self):
        return self.clone(GetYoutubeVideoIds)

    def output(self):

        # Returns the target file path for the video Ids
        return SuffixPreservingLocalTarget(
            os.path.join(
                "%s" % self.CAPTIONS_ROOT, "%s" % self.video_captions_target_file
            )
        )

    def run(self):

        # Reads in the video Ids from the required task and reads into a list
        data = []
        with self.input().open(mode="r") as input_target:
            for line in input_target.readlines():
                current_video = line[:-1]
                data.append(current_video)

        # Retrieves the captions for the above video Ids and writes
        # them to the output file path
        with self.output().open(mode="w") as output_target:
            output_target.write(write_to_file(data))
