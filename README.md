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
| Library | Use(s) |
|-|-|
| csci_utils (Our very own library): |*Provides access to functionality to preserve file extensions when writing to local targets within luigi tasks and for the of atomically writing to files* |
| nltk (Natural Language Toolkit): | *Provides functionality to preprocess text data (ex: checking if a word is English)* |
| genism: | *Provides access to the Word2Vec model architecture from which the word embeddings will be constructed* |
| youtube-transcripts-api: | *Provides an API from which to retrieve YouTube video caption data* |
| selenium: | *Provides advanced controlling of web browser functionality and retrieval of HTML elements (especially superior to BeautifulSoup when certain HTML elements are dynamically loaded via JavaScript)* |
| atomicwrites: | *Also used in parts of the code to ensure atomic writing to files* |


#### Data Acquisition
So, with the goal of accomplishing the above, our journey begins with the most fundamental requirement: Acquiring the captions. For this, we use the selenium package, which is succinctly described on pypi.org: “The selenium package is used to automate web browser interaction from Python.”

Roughly following the instructions provided by this source: https://www.analyticsvidhya.com/blog/2019/05/scraping-classifying-youtube-video-data-python-selenium/, I managed to produce the ‘get_video_ids’ function in  the ‘transcripts/captions.py.’
The first part of this function formats a YouTube url that includes the formatted version of the YouTube query that is one of the command-line arguments the user would enter on the command line. 

Throughout the course of this document, I will expand on the full command-line argument specifications the user will enter. We will build up to the full list of arguments one at a time. The first one just mentioned is the actual query that will be entered into the YouTube search bar:

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
1.	Open up an instance of the Chrome web browser

2.	Navigate to the formatted YouTube url
        - This will point to the search results for the query the user specified

3.	Scroll the page downwards
        - The amount by which to scroll downwards is also specified by the user
        - This amount is in units of pixels, ex: ``` -s 1000```
        - We want to scroll the page downwards because this will load more search results, thereby allowing us to retrieve more video captions

4.	Repeat step three a specified number of times
        - The number of times the downwards scrolling happens is also specified on the command-line, ex: ```-c 5```
        - Between each scroll, we wait five seconds so that the results have time to fully load
        - Once we have scrolled downwards (a specified pixel amount) a specified number of times, we need to extract an HTML element containing the video title
        - We do this for each video that has been loaded

6.	We extract the ‘href’ attribute of these HTML elements and create a set from them
        - Another argument that user will specify on the command-line will be an argument abbreviated as ‘-ch’ which represents the YouTube channel for a video. Although it should not break the program, the user should enter either ‘by cnn’ or ‘by fox’ as the value for this argument, ex: ```-ch ‘by cnn’```
        - This argument is useful because if we want to retrieve videos from CNN commentary about a particular political topic, we need some way to ensure the returned results are indeed from CNN and not another channel. Before we actually extract the ‘href’ element from one of our collected HTML elements, we check the ‘aria-label’ which should contain a string that contains the substring ‘by [channel_name]’.
        - If the ‘aria-label’ attribute of the HTML element contains, in the case of CNN video results, the string ‘by cnn’, we add it to the final set of video ids.

7.	Now we have a set of ‘href’ attributes, the final part of which contains the video ids
        - We subset this string to only retrieve the relevant video ids component, which looks something like ‘v=DEq5k_UvDB8’. 
        - Go ahead and check out the YouTube video id provided just above if you like watching Rhett and Link from Good Mythical Morning eat things (their banter is currently serving as the background soundtrack to this writeup)

8.	Finally, this set of video ids is returned to the calling function
        - I used the set data structure instead of a list because this helps enforce the use of each video only once.




```python
def get_video_ids(query_string, scroll_amount, cycles_to_scroll, uploader):
    ...
```


```python
def write_to_file(video_ids):
    ...
```


```python
def gen_sample_data(sample_list):
    ...
```


```python
def replace_chars(func, chars_to_remove=("'", ">", "-")):
    ...
```


```python
@replace_chars
def check_gibberish(data_to_check):
    ...
```


```python
def is_english(word_to_check):
    ...
```

```python
def remove_punctuation(list_to_filter):
    ...
```


```python
def remove_stop_words(list_to_filter):
    ...
```


```python
def construct_embedding(captions, cbow=True):
    ...
```


```python
def compile_all_captions(root_dir, news_organization="cnn"):
    ...
```


##### Advanced Python for Data Science Principles Used in this Project (in no particular order)
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


##### Future Improvements
1.	Make the code browser-agnostic and allow for this to be a specifiable option for the user (this one is probably a couple lines of code changes)
2.	Generalize the program to be able to compare any user chosen contexts (not just ‘CNN’ and ‘Fox News’)
3.	Perform certain steps concurrently rather than iteratively
        - Currently, the program accesses YouTube twice in order to retrieve video captions. Once to retrieve a list of video ids and then once to retrieve the associated captions for the subset of these videos that survived a certain filter
        - Instead, the program could be refactored to perform the filter at the same as it is obtaining the video ids themselves such that it would also be able to retrieve the captions in this first access of YouTube.
4.	Allow the user to choose the hyperparameters for the neural network
