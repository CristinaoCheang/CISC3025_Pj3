#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# --------------------------------------------------
# Description:
# --------------------------------------------------
# Author: Konfido <konfido.du@outlook.com>
# Created Date : April 4th 2020, 17:45:05
# Last Modified: April 4th 2020, 17:45:05
# --------------------------------------------------
import re

from nltk.classify.maxent import MaxentClassifier
from sklearn.metrics import (accuracy_score, fbeta_score, precision_score,
                             recall_score)
import os
import pickle
import nltk

nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# create a predictor class
def predict_sentence(sentence, classifier, MEMM, self=None):
    # remove the punctuation in words but not remove the Mr. Mrs. etc.
    pattern = r'\b(?:Mr|Mrs|Ms|Dr)[.]'
    abbreviations = re.findall(pattern, sentence)
    sentence = re.sub(pattern, 'ABBREVIATION', sentence)
    sentence = re.sub(r'[^\w\s]|_', '', sentence)
    for i, abb in enumerate(abbreviations):
        sentence = sentence.replace('ABBREVIATION', abb, 1)
    # Tokenize the sentence into words
    words = nltk.word_tokenize(sentence)
    prefixes = ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Rev.", "Hon."]
    suffixes = ["Jr", "Sr", "II", "III", "IV", "Esq."]
    i = 1
    k = 0
    while i <= len(words):
        if i >= 1 & i <= len(words):
            previous_word = words[i - 1]
            if previous_word in prefixes:
                words[i] = previous_word + words[i]
                words.remove(previous_word)
        i += 1

    while k < len(words) - 1:
        if k >= 0 & k <= len(words):
            next_word = words[k + 1]
            if next_word in suffixes:
                words[k] = words[k] + " " + next_word
                words.remove(next_word)
        k += 1
    # Initialize the previous label as "O"
    previous_label = "O"

    # Initialize an empty list to store the predicted labels
    predicted_labels = []

    # Loop through each word in the sentence
    for i in range(len(words)):
        # Extract the features for the current word
        features = MEMM.features(self, words, previous_label, i)

        # Use the classifier to predict the label for the current word
        predicted_label = classifier.classify(features)

        # Append the predicted label to the list
        predicted_labels.append(predicted_label)

        # Update the previous label for the next iteration
        previous_label = predicted_label

    # Return the list of predicted labels
    final_labels = []
    final_output = []
    for i in range(len(predicted_labels)):
        final_labels.append(words[i] + ' : ' + predicted_labels[i])
    return final_labels


class MEMM():

    def __init__(self):
        self.train_path = "../data/train"
        self.dev_path = "../data/dev"
        self.beta = 0
        self.max_iter = 0
        self.classifier = None

    def features(self, words, previous_label, position):
        """
        Note: The previous label of current word is the only visible label.

        :param words: a list of the words in the entire corpus
        :param previous_label: the label for position-1 (or O if it's the start
                of a new sentence)
        :param position: the word you are adding features for
        """

        features = {}
        """ Baseline Features """
        current_word = words[position]
        features['has_(%s)' % current_word] = 1
        features['prev_label'] = previous_label

        # ===== TODO: Add your features here =======#
        # Previous and next word
        prefixes = ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Rev.", "Hon."]
        suffixes = ["Jr", "Sr", "II", "III", "IV", "Esq."]

        if current_word[:3] in prefixes:
            features['Honorific'] = 1

        if current_word[-3:] in suffixes:
            features['Junior'] = 1

        # Word shape
        if current_word[0].isupper():
            features['Titlecase'] = 1
        elif current_word[0].islower():
            features['Lowercase'] = 1
        else:
            features['Mixcase'] = 1

        # =============== TODO: Done ================ #
        return features

    def load_data(self, filename):
        words = []
        labels = []
        for line in open(filename, "r", encoding="utf-8"):
            doublet = line.strip().split("\t")
            if len(doublet) < 2:  # remove emtpy lines
                continue
            words.append(doublet[0])
            labels.append(doublet[1])
        return words, labels

    def train(self):
        print('Training classifier...')

        words, labels = self.load_data(self.train_path)

        previous_labels = ["O"] + labels
        features = [self.features(words, previous_labels[i], i)
                    for i in range(len(words))]
        train_samples = [(f, l) for (f, l) in zip(features, labels)]
        classifier = MaxentClassifier.train(train_samples, max_iter=self.max_iter)

        self.classifier = classifier

    def test(self):
        print('Testing classifier...')
        words, labels = self.load_data(self.dev_path)
        previous_labels = ["O"] + labels
        features = [self.features(words, previous_labels[i], i)
                    for i in range(len(words))]
        results = [self.classifier.classify(n) for n in features]

        f_score = fbeta_score(labels, results, average='macro', beta=self.beta)
        precision = precision_score(labels, results, average='macro')
        recall = recall_score(labels, results, average='macro')
        accuracy = accuracy_score(labels, results)

        print("%-15s %.4f\n%-15s %.4f\n%-15s %.4f\n%-15s %.4f\n" %
              ("f_score=", f_score, "accuracy=", accuracy, "recall=", recall,
               "precision=", precision))

        return True

    def show_samples(self, bound):
        """Show some sample probability distributions.
        """
        words, labels = self.load_data(self.train_path)
        previous_labels = ["O"] + labels
        features = [self.features(words, previous_labels[i], i)
                    for i in range(len(words))]
        (m, n) = bound
        pdists = self.classifier.prob_classify_many(features[m:n])

        print('  Words          P(PERSON)  P(O)\n' + '-' * 40)
        for (word, label, pdist) in list(zip(words, labels, pdists))[m:n]:
            if label == 'PERSON':
                fmt = '  %-15s *%6.4f   %6.4f'
            else:
                fmt = '  %-15s  %6.4f  *%6.4f'
            print(fmt % (word, pdist.prob('PERSON'), pdist.prob('O')))

    def dump_model(self):
        with open('../model.pkl', 'wb') as f:
            pickle.dump(self.classifier, f)

    def load_model(self):
        with open('../model.pkl', 'rb') as f:
            self.classifier = pickle.load(f)
