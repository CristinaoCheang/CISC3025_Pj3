import nltk
from StartCodes.NER.MEM import predict_sentence
from django.http import JsonResponse
from django.shortcuts import render
from StartCodes.NER.MEM import MEMM
from StartCodes.NER import MEM
import pickle
from django.shortcuts import render


def classify_sentence(request, self=None):  # classify the sentence
    if request.method == 'POST':
        sentence = request.POST['sentence']  # get the sentence from the form
        classifier = nltk.data.load('./StartCodes/model.pkl', format='pickle')  # load the classifier
        prediction = predict_sentence(sentence, classifier, MEMM, self)  # predict the sentence
        return render(request, 'Output.html',
                      {'sentence': sentence, 'classification_result': prediction})  # render the output page
    else:
        return render(request, 'Input.html')  # render the input page
