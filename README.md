# Homework 3 - Which book would you recomend?

<p align="center">
  <img src="./images/logo.png" alt="Sublime's custom image"/>
</p>

# Task
The goal of this project was to experiment with crawling, parsing and to get confidence with different techniques regarding search engines, like how to retrieve results for a conjunctive query, how to score results with cosin similarity and tfidf, and so on so forth. Finally, it was asked to write both a recursive and a dynamic programming algorithm for the problem of longest increasing subsequence of a string.

We also decided to produce an original logo for our search engine! It does remind me of something, not sure what though...

> **Pronunciation**: goo·gs


# Usage
In the repository, it is included `requirements.txt`, which consists in a file containing the list of items to be installed using conda, like so:

`conda install --file requirements.txt`

Once the requirements are installed, you shouldn't have any problem when executing the scripts. Consider also creating a new environment, so that you don't have to worry about what is really needed and what not after you're done with this project. With conda, that's easily done with the following command:

`conda create --name <env> --file requirements.txt`

where you have to replace `<env>` with the name you want to give to the new environment.


# Repo structure
The repository consists of the following files:
1. __`data`__:
    > This directory contains both the data retrieved just after the crawling part (`parsed_books.tsv`) and the data after the preprocessing and cleaning part (`clean_data.csv`).
2. __`images`__:
    > This directory contains images for the search engine logo and for part of the recursive complexity proof. Just ignore this.
3.  __`indexes`__:
    > This directory contains the pickle objects for the vocabulary, the inverted index dictionary and the tfidf inverted index.
4. __`book_links.txt`__:
    > A txt file containing the links for all the html urls.
5. __`data_collector.py`__:
    > A Python script containing the functions to download the txt file and the html pages for the books.
6. __`functions.py`__:
     > A Python script containing all the functions used in the `main.ipynb`, apart from the data collection and parsing parts.
7. __`main.ipynb`__: 
    > A Jupyter notebook which provides the solutions to all the homework questions. The notebook just contains the answers; the only code provided here is the one for exercise 5, for which the answer is actually the code itself.
8. __`parser.py`__:
    > A Python script containing the functions to parse the html pages and extract the tsv file.
9. __`requirements.txt`__:
    > A txt file containing the dependecies of the project; see the usage part for details.