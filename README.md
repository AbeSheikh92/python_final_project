# 2020fa-final-project-AbeSheikh92

[![Build Status](https://travis-ci.com/csci-e-29/2020fa-final-project-AbeSheikh92.svg?token=sdLPZkWGsh3csqMrhXgK&branch=master)](https://travis-ci.com/csci-e-29/2020fa-final-project-AbeSheikh92)

[![Maintainability](https://api.codeclimate.com/v1/badges/f0df51fab4af19e09378/maintainability)](https://codeclimate.com/repos/5fcf139776007c01770081cd/maintainability)

[![Test Coverage](https://api.codeclimate.com/v1/badges/f0df51fab4af19e09378/test_coverage)](https://codeclimate.com/repos/5fcf139776007c01770081cd/test_coverage)
	

#### VERSION 2
##### Libraries Used:
| Library | Use(s) |
|-|-|
| csci_utils (Our very own library): |*Provides access to functionality to preserve file extensions when writing to local targets within luigi tasks* *Allows for the of atomic writing to files* |
| nltk (Natural Language Toolkit): | *Provides functionality to preprocess text data (ex: checking if a word is English)* |
| genism: | *Provides access to the Word2Vec model architecture from which the word embeddings will be constructed* |
| youtube-transcripts-api: | *Provides an API from which to retrieve YouTube video caption data* |
| selenium: | *Provides advanced controlling of web browser functionality and retrieval of HTML elements (especially superior to BeautifulSoup when certain HTML elements are dynamically loaded via JavaScript)* |
| atomicwrites: | *Also used in parts of the code to ensure atomic writing to files* |


### VERSION 2
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