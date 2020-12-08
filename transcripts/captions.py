import os
import time
from selenium import webdriver
from youtube_transcript_api import YouTubeTranscriptApi
from transcripts.data_cleaning import gen_sample_data, check_gibberish


def get_video_ids(query_string, scroll_amount, cycles_to_scroll, uploader):
    """
    Retrives a subset of video Ids from the loaded YouTube page
    related to the search query

    :param query_string: str representing YouTube search query
    :param scroll_amount: int representing the number of pixels to scroll the search results downwards by
    :param cycles_to_scroll: int representing the number of time to scrolls the search results downwards
    :param uploader: str representing the YouTube channel
    :return: unique set of videos Ids related to YouTube search query
    """

    # Formats the YouTube url to include the query
    # and filter for video with captions enabled
    # (which does NOT gaurantee captions are actually enabled)
    formatted_query = str(query_string).replace(" ", "+")
    link = (
        "https://www.youtube.com/results?search_query={0}&sp=EgIoAQ%253D%253D".format(
            formatted_query
        )
    )

    # 'driver' represents the 'controller' / 'manager' of the Chrome web browser
    driver = webdriver.Chrome()

    # The 'controller' accesses the link and scrolls an initial amount
    driver.get(link)
    driver.execute_script("window.scrollTo(0, {0});".format(scroll_amount))

    # Pauses 5 seconds between each downward scroll in order to allow the page to fully load
    # and thus better ensure html data will be available for acquisition
    pause_between_scrolls = 5.0

    # Lower bound on the scroll amount
    min_scroll_value = 0

    # The scrolling height is initially set to the 'scroll_amount'
    # which is later used as the scrolling increment amount
    scroll_height = scroll_amount

    # Scrolls downwards 'cycles_to_scroll' number of times
    for i in range(cycles_to_scroll):

        # The 'controller' executes a script to scroll downwards by
        # the specified amount
        driver.execute_script(
            "window.scrollTo({0}, {1});".format(min_scroll_value, scroll_height)
        )

        # Pauses activity for 5 seconds
        time.sleep(pause_between_scrolls)

        # Scroll bounds are updated
        min_scroll_value = scroll_height
        scroll_height += scroll_amount

    # Retrieves the video title elements
    user_data = driver.find_elements_by_xpath('//*[@id="video-title"]')

    # Set comprehension storing the corresponding full video Ids links if they are
    # uploaded by the intended YouTube channel as represented by 'uploader'.
    # The uploader is checked because any given search result page will have
    # a number of different YouTube channel results even if a specific one
    # was searched for. This ensures video Ids are only kept if they are
    # from the target channel.
    links = {
        i.get_attribute("href")
        for i in user_data
        if uploader in str(i.get_attribute("aria-label")).lower()
    }

    # Set comprehension storing the extracted video Id component from above
    video_id_set = {
        url_link[(url_link.index("v=") + len("v=")) :] for url_link in links
    }

    return video_id_set


def write_to_file(video_ids):
    """
    Concatenates all the caption data into a single string and
    direclty precedes the writing of this data to file

    :param video_ids: list of video Ids from which to extract captions
    :return: concatenated string of all caption text
    """

    total_data = ""
    for video in video_ids:
        try:

            # Retrieves the captions corresponding to the video Ids and
            # specifies the english translation
            captions_list = YouTubeTranscriptApi.get_transcript(video, languages=["en"])

            # Tests a sample of the caption data to ensure it is quality data
            sample_data = gen_sample_data(captions_list)
            gibberish = check_gibberish(sample_data)

            # If the sample data from the YouTube captions
            # did not pass the gibberish test, the video is skipped
            if gibberish:
                continue

            # Accumulates the text
            data = " ".join(list(map(lambda x: "".join(x["text"]), captions_list)))
            total_data += " " + data

        # Some common error types are specifically allowed to be displayed
        except (
            TypeError,
            ValueError,
            AttributeError,
            FileExistsError,
            FileNotFoundError,
        ) as e:
            print(e)
        except:
            print("Video {0} Does Not Have Captions Enabled".format(video))

    return total_data


def compile_all_captions(root_dir, news_organization="cnn", testing=False):
    """
    Compiles all of the target text files into a single text file

    :param root_dir: file path representing the root directory where the text files live
    :param news_organization: str representing a filter by which to search through the files
    :param testing: boolean representing which output file path should be used. This is done so
           that the path created during testing does not overwrite and/or interfere with
           the path created during regular execution of this program
    :return: file path of output file
    """

    # Searches through the list of text files and retrives only the ones matching
    # the intended news organization and those that have caption data
    file_list = [
        os.path.join(root_dir, file)
        for file in os.listdir(root_dir)
        if os.path.isfile(os.path.join(root_dir, file))
        and "captions" in file
        and news_organization in file
    ]

    # The output file path is altered if this execution is for testing
    if testing:
        target_path = os.path.join(os.path.abspath("data"), "test.txt")
    else:
        target_path = os.path.join(
            root_dir, "{0}_ALL_CAPTIONS.txt".format(news_organization)
        )

    # The text files are combined into one and the output file path is returned
    with open(target_path, mode="w+") as output_target:
        for file in file_list:
            with open(file, mode="r") as input_target:
                output_target.write(input_target.read())

    return target_path
