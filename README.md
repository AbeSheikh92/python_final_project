# 2020fa-final-project-AbeSheikh92

[![Build Status](https://travis-ci.com/csci-e-29/2020fa-final-project-AbeSheikh92.svg?token=sdLPZkWGsh3csqMrhXgK&branch=master)](https://travis-ci.com/csci-e-29/2020fa-final-project-AbeSheikh92)

[![Maintainability](https://api.codeclimate.com/v1/badges/f0df51fab4af19e09378/maintainability)](https://codeclimate.com/repos/5fcf139776007c01770081cd/maintainability)

[![Test Coverage](https://api.codeclimate.com/v1/badges/f0df51fab4af19e09378/test_coverage)](https://codeclimate.com/repos/5fcf139776007c01770081cd/test_coverage)

	
##### Motivation
In a previous semester I took a class in which I was initially introduced to the concept of word embeddings and their ability to learn vector representations for words based on the context in which these words were used. This basic idea of vectorizing meaning to the best of our current abilities was not in of itself the source of my interest. Instead, I thought what would be more interesting would be to compare the learned meaning/connotation of a set of words given the neural network is separately trained (so two different models) in two different contexts which create different word associations (as a consequence of different belief systems). 

##### Strategy
The specific contexts I chose were CNN and Fox News. It is my intuition that CNN and Fox News have drastically different views pertaining to a variety of issues applying to America and also the world in general. These differing views are conveyed on each news platform via language. And thus, encoded in the language is this difference in world views. The distinct encoding of views into language is accomplished through divergent word associations. As an example, if CNN had a world view in which cats were demonized as satanic death-worshipping hell-creatures, and Fox’s view encapsulated cats as paradisiacal love-puppets that sweat angel grace, this distinction in world views would certainly be conveyed in terms of the vocabulary used by each platform when speaking about cats. 

It is this difference in word associations that ultimately construct mostly non-overlapping connotations for cat-associated vocabulary. As such, I thought it would very interesting to capture a substantial portion of available YouTube caption data for both news organizations (after querying for a variety of controversial political topics), separately feed compiled text data to a Word2Vec neural network in order to produce two trained models that can then be queried with various terms (given they exist in the constructed vocabulary of the model) which would then return a list of N words whose vector representations are most similar to the query itself. Finally, I would feed the results into my optic nerves such that photonic delivery carries them to my central nervous system, which would then attempt to spot noticeable differences in the returned lists (one from a CNN trained model and one trained on Fox). There are certainly more ways by which to query the trained Word2Vec model in order to ascertain different insights, but I chose this for the sake of simplicity. 


#### A Brief Introduction to Borrowed Code
##### Libraries Used:
| Library | Use in This Project |
|-|-|
| csci_utils (Our very own library): |*Provides access to functionality to preserve file extensions when writing to local targets within luigi tasks and for the of atomically writing to files* |
| nltk (Natural Language Toolkit): | *Provides functionality to preprocess text data (ex: checking if a word is English)* |
| genism: | *Provides access to the Word2Vec model architecture from which the word embeddings will be constructed* |
| youtube-transcripts-api: | *Provides an API from which to retrieve YouTube video caption data* |
| selenium: | *Provides advanced controlling of web browser functionality and retrieval of HTML elements (especially superior to BeautifulSoup when certain HTML elements are dynamically loaded via JavaScript)* |
| atomicwrites: | *Also used in parts of the code to ensure atomic writing to files* |


#### Quick Start
Because the following detailed explanation is quite lengthy, this section will briefly explain how to quickly execute part 
of the project's functionality in order to prove that is does indeed do something besides exist. Once you have cloned the repo
locally, you can run the makefile provided in order to retrieve some pre-trained Word2Vec models. 

This should be doable on the command-line within the local project directory via:
* ```python3 make data```.
Once you've downloaded the models, one of which is trained on YouTube video caption data from CNN and one of which is trained
on YouTube caption data from Fox News, you can run the following on the command-line:
* ```python3 -m transcripts -n 'fox' -sq 'biden'```
* ```python3 -m transcripts -n 'fox' -sq 'trump'```
* ```python3 -m transcripts -n 'fox' -sq 'obama'```
* ```python3 -m transcripts -n 'cnn' -sq 'biden'```
* ```python3 -m transcripts -n 'cnn' -sq 'trump'```
* ```python3 -m transcripts -n 'cnn' -sq 'obama'```
    
At this point, under the:
* data/models/ALL_CAPTIONS_CNN/queries
* data/models/ALL_CAPTIONS_FOX/queries

directories each, you should find csv files entitled:
* biden_model_results.csv
* trump_model_results.csv
* obama_model_results.csv

Go ahead and compare the corresponding results from both models. The following is my attempt to tl;dr what the results mean:
* A good bit of YouTube video caption data was retrieved from a variety of CNN videos that were returned given a variety of 
searches for politically controversial topics (guns, immigration, etc.).
* This caption text data was preprocessed and then fed to a Word2Vec model which creates word embeddings.
* This trained model is one of the trained models you just downloaded.
* The other trained model was identically created but with Fox News YouTube video caption data.
* When you type ```python3 -m transcripts -n 'fox' -sq 'biden'```, this will execute the following python code:
    * ```python some_trained_model.most_similar('biden')```
    * The __most_similar__ method will ask the trained model which words it has determined are most likely to appear within 
    the context of the input term, which in this case, is 'biden'.
* The results will be 30 such words listed from highest to lowest in terms of likelihood of appearing in the context of the 
input word ('biden'). 

Go ahead and compare the results from the model results you have already created and try creating some more.

Now, for the detailed version... read on. 


#### Data Acquisition
So, with the goal of accomplishing the above, our journey begins with the most fundamental requirement: Acquiring the captions. For this, we use the selenium package, which is succinctly described on pypi.org: “The selenium package is used to automate web browser interaction from Python.”

Roughly following the instructions provided by this source: https://www.analyticsvidhya.com/blog/2019/05/scraping-classifying-youtube-video-data-python-selenium/, I managed to produce the ‘get_video_ids’ function in  the ‘transcripts/captions.py.’
The first part of this function formats a YouTube url that includes the formatted version of the YouTube query that is one of the command-line arguments the user would enter on the command line. 

Throughout the course of this document, I will expand on the full command-line argument specifications the user will enter. We will build up to the full list of arguments one at a time. The first one just mentioned represents the actual query that will be entered into the YouTube search bar:

Ex:
```bash
-q 'cnn immigration'
```

The first part of the ‘get_video_ids’ function constructs a formatted YouTube url including the query and filters for those videos that have captions enabled. Still, this does not guarantee that the filtered video will all have captions enabled.

```python
def get_video_ids(query_string, scroll_amount, cycles_to_scroll, uploader):

    formatted_query = str(query_string).replace(" ", "+")
    link = (
        "https://www.youtube.com/results?search_query={0}&sp=EgIoAQ%253D%253D".format(
            formatted_query
        )
    )
    ...
```

The next set of commands can be briefly summarized as using selenium to control the web browser (which for the scope of this project is currently specified as Chrome). The following are the sequential steps we want to be able to get the Chrome web browser to do:
* Open up an instance of the Chrome web browser
```python
    # 'driver' represents the 'controller' / 'manager' of the Chrome web browser
    driver = webdriver.Chrome()
    ...
```

* Navigate to the formatted YouTube url
    * This will point to the search results for the query the user specified
```python
    ...

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
    ...
```

* Scroll the page downwards
    * The amount by which to scroll downwards is also specified by the user
    * This amount is in units of pixels
        * Ex: ``` -s 1000```
    * We want to scroll the page downwards because this will load more search results, thereby allowing us to retrieve more video captions
    * Repeat the above step a specified number of times
        * The number of times the downwards scrolling happens is also specified on the command-line
            * Ex: ```-c 5```
        * Between each scroll, we wait five seconds so that the results have time to fully load
        * Once we have scrolled downwards (a specified pixel amount) a specified number of times, we need to extract an HTML element containing the video title
        * We do this for each video that has been loaded
```python
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
    ...
```

* We extract the ‘href’ attribute of these HTML elements and create a set from them
    * Another argument that user will specify on the command-line will be an argument abbreviated as ‘-ch’ which represents the YouTube channel for a video. Although it should not break the program, the user should enter either ‘cnn’ or ‘fox’ as the value for this argument.
        * Ex: ```-ch 'cnn'```
    * This argument is useful because if we want to retrieve videos from CNN commentary about a particular political topic, we need some way to ensure the returned results are indeed from CNN and not another channel. Before we actually extract the ‘href’ element from one of our collected HTML elements, we check the ‘aria-label’ which should contain a string that contains the substring ‘by [channel_name]’.
    * If the ‘aria-label’ attribute of the HTML element contains, in the case of CNN video results, the string ‘by cnn’, we add it to the final set of video ids.
```python
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
    ...
```
* Now we have a set of ‘href’ attributes, the final part of which contains the video ids
    * We subset this string to only retrieve the relevant video ids component, which looks something like ‘v=DEq5k_UvDB8’. 
    * Go ahead and check out the YouTube video id provided just above if you like watching Rhett and Link from Good Mythical Morning eat things (their banter is currently serving as the background soundtrack to this writeup)
```python
    # Set comprehension storing the extracted video Id component from above
    video_id_set = {
        url_link[(url_link.index("v=") + len("v=")) :] for url_link in links
    }
    ...
```
* Finally, this set of video ids is returned to the calling function
    * I used the set data structure instead of a list because this helps enforce the use of each video only once.
```python
    return video_id_set
```

At this point, we now have set of YouTube video ids from which subsequent code will extract captions from. The function designed
to do this is the __write_to_file__ function. Essentially, this function is sent a list of video ids and cycles through them,
retrieving each video's corresponding captions. 

```python
def write_to_file(video_ids):

    total_data = ""
    for video in video_ids:
        try:

            # Retrieves the captions corresponding to the video Ids and
            # specifies the english translation
            captions_list = YouTubeTranscriptApi.get_transcript(video, languages=["en"])
            ...
            ...
            ...

            # Accumulates the text
            data = " ".join(list(map(lambda x: "".join(x["text"]), captions_list)))
            total_data += " " + data
        except:
            ...

    return total_data
```

However, once the captions for a video are retrieved, via the line reading,

```python
captions_list = YouTubeTranscriptApi.get_transcript(video, languages=["en"])
```

they need to be preprocessed. The specific need for them to have this preprocessing is evident due to certain peculiarities 
that sometimes express themselves. Basically, sometimes the captions are returned as either non-English or gibberish. Even though
the call to the YouTube API specifies the English captions, they are not always returned as such. Also, sometimes the captions
seem to be poorly generated, and thus most of the words are gibberish. For this reason, before a particular video's captions
are added to the text that will be used to train an instance of a Word2Vec model, the video's captions must be checked to ensure
its contents will be useful to the model and not dilutive. 

We will go over the four star functions that will help us in this process. They are:
* gen_sample_data
* check_gibberish
* replace_chars
* is_english

It will surely be more time-consuming to check each and every word of a video's returns captions list, and so perhaps choosing
a sample of words from each video's captions would serve to identify which video should not be used while also minimizing the
computational and time complexity required to do this.

Selecting 10 consecutive words from each video's captions should do well for this task. And below we can see the __gen_sample_data__
function doing exactly that:

```python
def gen_sample_data(sample_list):

    sample_data = [sample_list[i]["text"] for i in range(len(sample_list)) if i < 10]
    return sample_data
```

Once __gen_sample_data__ has selected 10 words from a given video's captions, these words are then passed to the __check_gibberish__
function. 

The __check_gibberish__ function essentially receives a tokenized list of strings and passes each element (each of which is a word) of this list to the 
__is_english__ function (after selectively filtering for which words would be best to send). The nltk corpus library can sometimes
have issues identifying words that end with an 's' when the 's' indicates plurality. For this reason, __check_gibberish__ filters
out these words. But before __check_gibberish__ does any of this, the incoming tokenized list of strings is preprocessed first.

Enter, __replace_chars__. The __replace_chars__ function decorates __check_gibberish__. It is within this wrapper that certain 
characters are filtered from the data that __gen_sample_data__ is sending __check_gibberish__. The reason this filtering is necessary,
is that some of __gen_sample_data__'s 10 selected words may have (due to caption data peculiarities) certain characters in them
that would confuse the nltk corpus library. Some examples may look like:
* '>>>phone' (needs to be 'phone')
* 'exit.' (needs to be 'exit')
* 'won't' (needs to be 'wont')

After these characters are removed from the words __gen_sample_data__ is sending to __check_gibberish__ (which are being intercepted
by this decorator), they are re-tokenized because in the character removal process the list of strings was transformed into 
a single string. For this reason, before this data is allowed to pass to __check_gibberish__, it needs to be transformed back into 
a list of strings where each element is a single word. 

```python
def replace_chars(func, chars_to_remove=("'", ">", "-")):

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

```

Once __replace_chars__ has preprocessed the data, the decorator lets the data pass, in its new form, to __check_gibberish__.
As mentioned above, __check_gibberish__ filters out some words (if need be) and sends them, one at a time, to the fourth function, 
__is_english__. 

```python
@replace_chars
def check_gibberish(data_to_check):

    check_list = [
        is_english(word.lower()) * 1
        for word in data_to_check
        if len(word) > 1 and word[-1] != "s"
    ]

    # try different values for the threshold
    return sum(check_list) / float(len(check_list)) < 0.80
```

The __is_english__ function simply asks the nltk.corpus.words package if each word that is sent to it is indeed an English word. 
It either returns True or False.

```python
def is_english(word_to_check):

    if word_to_check in words.words():
        return True
    else:
        return False
```

One final function the __check_gibberish__ function performs, is calculating what fraction of the words (for this single 10-word sample
for one video's captions) were indicated to be English out of all the words it sent to __is_english__. If this fraction is at least
80%, check_gibberish returns True to its calling function, and False otherwise. The reason check_gibberish does not ensure 
that 100% of the words must be identified to be English is for a couple reasons:
* the nltk.corpus.words package will not always correctly identify English words as English
* other erroneous processes/details may cause some English words to be misidentified as non-English

For these reasons, a 20% gap is allocated for just these sort of errors. Recall, that the __write_to_file__ function is cycling 
through each video id sent to it, and doing the following:
* retrieving its YouTube caption data
* creating a 10-word sample of data from these captions
* checking if at least 80% of these captions are usable
    * if so, add these captions to a string (that is to be concatenated with subsequent video captions)
    * if not, ignore the video's captions
    
This is illustrated with the full __write_to_file__ function depicted below:
```python
def write_to_file(video_ids):

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
```
 
To reiterate, what __write_to_file__ returns, is a single string that is the accumulated concatenation of all the caption data from 
all the video ids sent to it. This string is what we will use to train the Word2Vec model. This is done via our next key
function, __construct_embedding__:

This function performs some of the preprocessing on this single string that was performed on only the sample data discussed previously.
This preprocessing includes:
* remove certain characters from the string ('\n', '>', '--')
* tokenize the string into a list where each element contains one word from the string
* removing punctuation 
* removing stop words 
* remove single-character words/elements

Certain hyper-parameters associated with the Word2Vec model are hard-coded, including:
* type of model (CBOW / Skip-gram)
* min_count
* vector_size
* window_size

Please see the following link for specifics pertaining to the above hyper-parameters:
https://radimrehurek.com/gensim/models/word2vec.html

```python
def construct_embedding(captions, cbow=True):

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
```

For completeness, the code to remove punctuation and stop words is shown below: 
```python
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
```

```python
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
```

At this point, we have gone over the heart of the code which makes our goal, achievable. Also, just to recap, the command-line 
usage of this program so far is achievable with an examples like the below:
* ```python3 -m transcripts -ch 'cnn' -q 'immigration' -s 20000 -c 5```
* ```python3 -m transcripts -ch 'fox' -q 'obama -s 15000```
* ```python3 -m transcripts -ch 'cnn' -q 'russia```

Things to note:
* The ```python3 -m transcripts``` executes __main.py__ from within the transcripts folder as a module.
* The command-line arguments that should not be left out, even though defaults exist, are:
    * -ch (signifying the YouTube channel uploader)
    * -q (signifying the term to be searched for on YouTube)
        * Although I minimally tested multi-term queries and the program worked, I would recommend single-word queries
        in order to ensure no magical bugs appear
        

In the __transcripts/tasks__ directory, we find 4 luigi tasks. These tasks in of themselves do not contribute any new substantial
code and/or functionality. Instead, they serve more-so to provide managerial structure to the functions described above such that
these functions will be executed in the proper order and only when need be. 

Let's briefly go over them. Firstly, in the __transcripts/tasks/caption_tasks.py__ file, we find __GetYouTubeVideos__. This task
receives the arguments entered on the command-line and assigns them to its internal parameters as follows:
* -ch -> channel_author parameter
* -q -> query parameter
* -s -> scroll_volume parameter
* -c -> num_cycles parameter

```python
class GetYoutubeVideoIds(ExternalTask):  # pragma: no cover
    """
    Manages the retrieval of YouTube video Ids related
    to the search query
    """

    # Parameter representing the local directory root
    LOCAL_ROOT = Parameter(default=os.path.abspath("data"))

    # Parameter representing the YouTube search query
    query = Parameter(default="immigration")

    # Parameter representing the number of pixels to
    # scroll the loaded YouTube page downwards
    scroll_volume = IntParameter(default=0)

    # Parameter representing the number of times to scroll
    # the screen downwards (each scroll scrolls downwards
    # by the 'scroll_volume' pixel amount)
    num_cycles = IntParameter(default=0)

    # Parameter representing the YouTube channel uploaded
    channel_author = Parameter(default="cnn")
```

This major undertaking of this task is in the __run__ method, which calls the __get_video_ids__ function discussed earlier,
and writes the returned video ids to a file whose name is constructed as such: [channel_author]_[query]_ids.txt'.
* For example, if the command-line arguments are: ```python3 -m transcripts -ch 'cnn' -q 'immigration'```, then the filename
would be cnn_immigration_ids.txt.
```python
    def output(self):
        # Represents the file path to store the video Ids
        video_ids_path = str(self.channel_author) + '_' + str(self.query) + '_ids.txt'

        # Returns the target file path for the video Ids
        return SuffixPreservingLocalTarget(
            os.path.join("%s" % self.LOCAL_ROOT, video_ids_path)
        )

    def run(self):
        # Retrieves video Ids and writes them to the target file path
        video_list = get_video_ids(
            str(self.channel_author) + " " + str(self.query),
            self.scroll_volume,
            self.num_cycles,
            self.channel_author
        )

        with self.output().open(mode="w") as output_target:
            for video in video_list:
                output_target.write("%s\n" % video)
```

A data folder is created by this project in which these video id text files are stored. 

Secondly, we have the __ProcessCaptionData__ task. This task basically does the following:
* Open the file where the video ids are saved
* Send a list of these ids to the __write_to_file__ function
* Write the returned concatenated string to a file whose name is constructed as follows: : [channel_author]_[query]_captions.txt.
* For example, if the command-line arguments are: ```python3 -m transcripts -ch 'cnn' -q 'immigration'```, then the filename
would be cnn_immigration_captions.txt.

```python
@inherits(GetYoutubeVideoIds)
class ProcessCaptionData(Task):  # pragma: no cover

    # Parameter representing the local directory root
    CAPTIONS_ROOT = Parameter(default=os.path.abspath("data"))

    def requires(self):
        return self.clone(GetYoutubeVideoIds)

    def output(self):
        # Represents the file path to store the video captions
        video_ids_path = str(self.channel_author) + '_' + str(self.query) + '_captions.txt'

        # Returns the target file path for the video Ids
        return SuffixPreservingLocalTarget(os.path.join("%s" % self.CAPTIONS_ROOT, video_ids_path))

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
```

Thirdly, in the __transcripts/tasks/model_results_tasks.py__ file, we have the __FeedToModel__ task. This task does the following:
* Opens up the file where the video captions are stored
* Sends this string to the __construct_embedding__ in order to create a Word2Vec model
* Writes the returned model to a file whose full path is constructed as such:
    * data/models/[channel_author]_[query]_captions/trained_embedding.model

```python
@inherits(ProcessCaptionData)
class FeedToModel(Task):  # pragma: no cover
    """
    Manages the feeding of the accumulated caption data to
    the Word2Vec word embedding model and thus the
    construction of the model
    """

    # Parameter representing the local file root
    MODEL_ROOT = Parameter(default=os.path.abspath(os.path.join("data", "models")))

    # Parameter representing the target file to save the model to
    trained_model_target_file = Parameter(default="trained_embedding.model")

    def requires(self):
        return self.clone(ProcessCaptionData)

    def output(self):

        # Returns the output target file path. The path incorporates
        # the captions target file path for a partial salting.

        # Represents the file path to store the video captions
        video_ids_path = str(self.channel_author) + '_' + str(self.query) + '_captions'

        paths_list = [
            str(self.MODEL_ROOT),
            video_ids_path,
            str(self.trained_model_target_file),
        ]

        return SuffixPreservingLocalTarget(reduce(os.path.join, paths_list))

    def run(self):

        # Reads in the caption data
        with self.input().open(mode="r") as input_target:
            captions_data = input_target.read()

        # Feeds data to and creates model
        trained_model = construct_embedding(captions=captions_data, cbow=True)

        # Saves model to target file
        with self.output().open(mode="w") as output_target:
            trained_model.save(output_target.name)
```

Fourthly, we have in the same file, the AnalyzeModelResults task. This task does the following:
* Opens the file where the trained model is stored
* Queries the model using the __most_similar__ method with a command-line argument abbreviated as '-sq'
    * Ex: ```-sq 'biden'```: command-line argument
    * Ex: trained_model.most_similar([term_of_interest]): usage of __most_similar__ method with the term entered on the command-line
* Write the 30 (hard-coded) returned words that the trained model determines are most likely (sorted from highest probability to 
lowest) to appear in the context of the word with which it has been queried
* Print the results of the model to screen

At this point, the full command-line specifications (including the less important '-s' and '-c' ones) are:
* Ex: ```python3 -m transcripts -ch 'cnn' -q 'immigration' -sq 'foreigners'```


```python
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

    # Parameter representing the model results target file
    model_results_target_file = Parameter(default="model_results.csv")

    # Parameter representing the term for which to retrieve
    # the top N similar words
    search_term = Parameter(default="news")

    def requires(self):
        return self.clone(FeedToModel)

    def output(self):

        # Represents the file path to store the video captions
        video_ids_path = str(self.channel_author) + '_' + str(self.query) + '_captions'

        paths_list = [
            str(self.RESULTS_ROOT),
            video_ids_path,
            str(self.model_results_target_file),
        ]

        return SuffixPreservingLocalTarget(reduce(os.path.join, paths_list))

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
```

It was at this point in the development, when I further realized that the model would be much better if it was trained on 
the collection of [channel_author]_[query]_captions.txt files I had at this point accumulated. Throughout the evolution of this
project I executed the project on a number of different controversial political topics, searching YouTube for both CNN and 
Fox News' responses. As such, any one caption text file was insignificant compared to the small corpus which all of them together
defined. 

Things to note:
* This task and the next one to be discussed are not designed such that they will trigger the four previous tasks. Although, it 
is the case that these will only produce any output if the four previous tasks have been used to produce at least one captions text
file for the news organization for which you will be executing the __CreateAccumulatedModel__ task.
* Let's assume the four previous tasks have been run and have produced a number of caption text files for Fox News. If you
now execute: ```python3 -m transcripts -n 'cnn'```, the next task to be discussed (__CreateAccumulatedModel__), will go into the __data__ directory,
compile all the files which contain 'cnn' and 'captions' in their filename into one large file by the name of 'cnn_ALL_CAPTIONS.txt'
within the same __data__ directory.  

Continuing the discussion, we now look to the __transcripts/tasks/assemble_tasks.py__ file, in which we find the __CreateAccumulatedModel__
task. This task does the following:
* Employ the __compile_all_captions__ function to find all the caption files in the __data__ directory that belong to the specified news 
organization (CNN or Fox News) and combine them into a new larger caption text file. The __compile_all_captions__ function 
shown below, searches the __data__ directory for all files that contain the word 'captions' and contain the name of the 
news organization specified by the command-line argument, -n. It then combines all their text into a single larger text 
file at the path of data/[news_organization]_ALL_CAPTIONS.txt. 

```python
def compile_all_captions(root_dir, news_organization="cnn"):

    # Searches through the list of text files and retrives only the ones matching
    # the intended news organization and those that have caption data
    file_list = [
        os.path.join(root_dir, file)
        for file in os.listdir(root_dir)
        if os.path.isfile(os.path.join(root_dir, file))
        and "captions" in file
        and news_organization in file
    ]

    target_path = os.path.join(root_dir, "{0}_ALL_CAPTIONS.txt".format(news_organization))

    # The text files are combined into one and the output file path is returned
    with open(target_path, mode="w+") as output_target:
        for file in file_list:
            with open(file, mode="r") as input_target:
                output_target.write(input_target.read())

    return target_path

```

* Send this data to the __construct_embedding__ function to train a new Word2Vec model (hopefully superior)
* Write the returned trained model to an output file who full path is constructed as such:
    * data/models/ALL_CAPTIONS_[news_organization]/trained_embedding.model
    * Ex: data/models/ALL_CAPTIONS_CNN/trained_embedding.model (in this case, this is for CNN)

```python
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
```

Now that we have the a new trained model that has 'learned' from a substantially larger corpus of text, we should query it.
This is where, in this same __transcripts/tasks/assemble_tasks.py__ file we find the __QueryUltraModel__ task. This task does 
the following:
* Opens the file where the newly trained ULTRA model is stored
* Query the model using the __most_similar__ method and an input term
* Write the returned list of words to an output file whose full path is constructed as such:
    * data/models/ALL_CAPTIONS_[news_organization]/queries/[model_input_term]_model_results.csv
    * For example, if we executed: ```python3 -m transcripts -ch 'cnn' -sq 'biden'```, this would produce the following path:
    data/models/ALL_CAPTIONS_CNN/queries/biden_model_results.csv
    
Things to note:
* Once the __CreateAccumulatedModel__ task has, for example, created the newly CNN-caption-trained model, anytime the
__QueryUltraModel__ task is executed, the __CreateAccumulatedModel__ task will not run since the output file already exists.
This is not really something to note since luigi is designed exactly to do this, I just thought I would point out that with 
each execution of __QueryUltraModel__, it certainly is NOT the case the the ULTRA model is having to be trained again and again, inefficiently.
* By having the single-term query be part of the actual filename, we achieve a sort of salting this way, ensuring that if
the same query is run on a pre-trained model, no tasks will be executed.
    * For example, running: ```python3 -m transcripts -ch 'cnn' -sq 'biden'``` will only execute the first time. However, since
    the '-sq' argument salts the output file name for this task, we CAN run any number of similar calls with different values
    for this argument which will create new output file entries under the data/models/ALL_CAPTIONS_CNN/queries directory. 
        * For example, if we run:
            * ```python3 -m transcripts -ch 'cnn' -sq 'trump```
            * ```python3 -m transcripts -ch 'cnn' -sq 'guns'```
        this produces the following full file paths:
            * data/models/ALL_CAPTIONS_CNN/queries/trump_model_results.csv
            * data/models/ALL_CAPTIONS_CNN/queries/guns_model_results.csv
            
```python
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
```

If you have made it to the end of this lengthy adventure, bless your heart. Only the strongest of wills can read through 
nearly 6,000 code-infused words and still retain their sanity.


#### Advanced Python for Data Science Principles Used in this Project (in no particular order)
1. Git work flow via utilization of development branches

2. Semantic versioning
3. Utilization of CSCI_UTILS library and thus enforcement of atomicity and proper preservation of file extensions
4. Luigi data pipeline
5. Proper file handling via context managers
7. Treating of functions as first-class variables via decorators
8. Interfacing with remote database structures such as AWS S3
9. Use of more advanced functional methodologies (lambda / map / reduce / Comprehensions)
10. Unit tests
11. Continuous integration / development via Travis / CC
12. Isolation via virtual environments (thanks to Pipenv)
13. Successful interfacing with third-part API


#### Future Improvements
1. Make the code browser-agnostic and allow for this to be a specifiable option for the user (this one is probably a couple lines of code changes)

2. Generalize the program to be able to compare any user chosen contexts (not just ‘CNN’ and ‘Fox News’)

3. Perform certain steps concurrently rather than iteratively
    3. Currently, the program accesses YouTube twice in order to retrieve video captions. Once to retrieve a list of video ids and then once to retrieve the associated captions for the subset of these videos that survived a certain filter
    3. Instead, the program could be refactored to perform the filter at the same as it is obtaining the video ids themselves such that it would also be able to retrieve the captions in this first access of YouTube.
    
4.	Allow the user to choose the hyperparameters for the neural network
