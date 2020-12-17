import argparse
from distutils.util import strtobool
from luigi import build
from .tasks.model_results_tasks import AnalyzeModelResults
from .tasks.assemble_task import QueryUltraModel


# Command line arguments specifying certain parameters for the tasks
parser = argparse.ArgumentParser(description="Arguments for Embedding Application")

parser.add_argument(
    "-u",
    "--ultra",
    default='False',
    help="Answers the question: 'Will you be entering commands to query the ultra model"
    "which is trained from the compilation of a small corpus of already created"
    "text files? If 'False', then you will be entering commands to create an individual"
    "video id text file and video caption text file and will be query a less ultra "
    "model that is trained off of only the caption data associated with this single "
    "caption text file.",
)

# Parameters for the tasks in caption_tasks.py and model_results_tasks.py
# -q represents the query to search for on YouTube
parser.add_argument("-q", "--query", default="immigration", help="Youtube Search Query")

# -ch represents the intended YouTube channel whose content is of interest
parser.add_argument("-ch", "--channel", default="cnn", help="Name of Youtube Channel")

# -s represents the number of pixels to scroll downwards on the loaded YouTube page
# in order to load more content
parser.add_argument(
    "-s",
    "--scroll",
    default=0,
    help="Amount of Pixels to be Scrolled For Each of the Scroll Cycles",
)

# -c represents the number of times to perform the above mentioned downwards scrolling
parser.add_argument("-c", "--cycles", default=0, help="Amount of Times to Scroll Down")

# -m represents the output target file path for the trained model
parser.add_argument(
    "-m",
    "--model_file",
    default="trained_embedding.model",
    help="Name of File to Save Model to",
)

# -mr represents the output target file path for the trained model results
parser.add_argument(
    "-mr",
    "--model_results",
    default="model_results.csv",
    help="Name of File to Save Model Results to",
)

# -sq represents the term to be input to the trained model in order for it
# to return the most similar words as computed during its training
parser.add_argument("-sq", "--search_query", default="news", help="Term of Interest")

# ---------------------------------------------------------------------------------------------------------------------
# Parameters for the task in assemble_task.py (-sq is also in the below task)
# -n represents the news organization whose captions are to be compiled
parser.add_argument(
    "-n",
    "--news",
    default="cnn",
    help="News Organization Whose Captions are to be Compiled",
)


"""
Note: Try not to mess with the computer while the 
code is running as it may disrupt the api calls 
(for some unknown reason) and result in no or 
little downloaded captions
"""


def main(args=None):
    args = parser.parse_args(args=args)
    ultra_task = bool(strtobool(args.ultra))

    if not ultra_task:
        build(
            [
                AnalyzeModelResults(
                    query=args.query,
                    scroll_volume=int(args.scroll),
                    num_cycles=int(args.cycles),
                    channel_author=args.channel,
                    search_term=args.search_query
                )
            ],
            local_scheduler=True,
        )

    else:
        build(
            [
                QueryUltraModel(
                    news_organization=args.news, term_of_interest=args.search_query
                )
            ],
            local_scheduler=True,
        )
