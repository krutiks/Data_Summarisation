from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from collections import defaultdict
import argparse
from heapq import nlargest

def __init__(self,filename):
    self.filename = filename

def main():
    read = read_file()
    filtered_words, tokenised_sentence = tokenize(read)
    rank = scoring(filtered_words,tokenised_sentence)
    return summurize(rank,tokenised_sentence,len(tokenised_sentence))

def parse_arguments():
    """ Parse command line arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='File name of text to summarize')
    parser.add_argument('-l', '--length', default=4, help='Number of sentences to return')
    args = parser.parse_args()

    return args

def read_file():
    text = open("somenotes.txt",encoding="utf-8")
    text = text.read()
    return text

def tokenize(text):
    word_tokens = word_tokenize(text)
    stop = set(stopwords.words('english'))
    tokenised_sentence = sent_tokenize(text)
    filtered_words = [i for i in word_tokens if str(i).lower() not in stop]
    return [filtered_words , tokenised_sentence]

def scoring(filtered_words,tokenised_sentence):
    words_Freq = FreqDist(filtered_words)
    ranking = defaultdict(int)

    for i,sentence in enumerate(tokenised_sentence):
        for word in word_tokenize(sentence.lower()):
            if word in words_Freq:
                ranking[i] += words_Freq[word]
    return ranking

def summurize(ranks,sentences,length):
    if 7 > len(sentences):
        print("Error, more sentences requested than available. Use --l (--length) flag to adjust.")
        exit()
    indexes = nlargest(7, ranks, key=ranks.get)
    final_sentences = [sentences[j] for j in sorted(indexes)]
    return '\n'.join(final_sentences)

if __name__ == "__main__":
    print(main())
