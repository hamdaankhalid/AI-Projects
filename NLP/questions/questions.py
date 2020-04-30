import operator
import string

import nltk
import sys
import os
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, file_idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    final = dict()
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r') as file:
            data = file.read().replace('\n', '')
            final[filename] = data

    return final


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    document = nltk.word_tokenize(document)
    document = [i.lower() for i in document if i not in nltk.corpus.stopwords.words('english')]
    document = [''.join(c for c in s if c not in string.punctuation) for s in document]
    document = [s for s in document if s]

    return document


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """

    """
    Inverse document frequency of a word is defined by taking the natural logarithm of the number of 
    documents divided by the number of documents in which the word appears.
    """

    idfs = dict()

    # for every document got through all the words in every document, and count its occurences
    docs = documents.keys()
    numerator = len(docs)

    #all words in all dictionaries:
    for name, doc in documents.items():

        for word in doc:
            word_doc = 0

            for name2, doc2 in documents.items():
                if word in doc2:
                    word_doc += 1

            idfs[word] = math.log(numerator/word_doc)

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    top = dict()
    for name, val in files.items():
        file_val = 0

        for word in query.intersection(set(val)):
            tf_idf = val.count(word) * idfs[word]
            file_val += tf_idf

        top[name] = file_val

    top = sorted(top, key=lambda k: (top[k]), reverse=True)[:n]

    return top


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    top_sent = dict()
    for sent, sent_list in sentences.items():
        sent_val = 0
        sent_den = len( query.intersection(set(sent_list))) / len(sentences)

        for word in query.intersection(set(sent_list)):
            sent_val += idfs[word]

        if sent_den > 0:
            top_sent[sent] = sent_val, sent_den

    top_sent = sorted(top_sent, key=lambda k: (top_sent[k]), reverse=True)[:n]
    return top_sent


if __name__ == "__main__":
    main()
