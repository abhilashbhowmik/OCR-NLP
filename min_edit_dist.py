# NC Mohit Lab-6 BT18GCS031
import nltk
from nltk.corpus import words
from nltk import FreqDist
from nltk.util import ngrams

corpus = words.words()

def min_edit_dist(word1,word2):
    matrix = []
    for i in range(len(word1)+1):
        row = []
        for i in range(len(word2)+1):
            row.append(0)
        matrix.append(row)
    for i in range(1,len(word1)+1):
        matrix[i][0] = i
    for j in range(1,len(word2)+1):
        matrix[0][j] = j
    for i in range(1,len(word1)+1):
        for j in range(1,len(word2)+1):
            val1 = matrix[i-1][j] + 1
            val2 = matrix[i][j-1] + 1
            if(word1[i-1]==word2[j-1]):
                val3 = matrix[i-1][j-1]
            else:
                val3 = matrix[i-1][j-1] + 2
            matrix[i][j] = min(val1,val2,val3)
    #printmatrix(matrix)
    return matrix[len(word1)][len(word2)]

def find_word(text):
    bigrams = ngrams(text, 2)
    fdist = nltk.FreqDist(bigrams)
    frequency_of_occurences = {}
    for word in corpus:
        counter = 0
        for bigram in fdist.items():
            bgram = bigram[0][0]+bigram[0][1]
            if(bgram in word):
                counter += 1
        frequency_of_occurences[word] = counter
    min_occurence_threshold = int(len(text)/2) + 1
    frequency_of_occurences = dict(sorted(frequency_of_occurences.items(), key=lambda item: item[1]))
    filtered_occurences = []
    for occur in frequency_of_occurences.items():
        if(occur[1] >= min_occurence_threshold):
            filtered_occurences.append([occur[0],occur[1]])
    answer = ""
    mindist = 9999
    for word in filtered_occurences:
        minimum_dist = min_edit_dist(text,word[0])
        if(minimum_dist < mindist):
            mindist = minimum_dist
            answer = word[0]
    if(answer == ""):
        return text
    return answer