import nltk

from StartCodes.NER.MEM import predict_sentence
from django.http import JsonResponse
from django.shortcuts import render
from StartCodes.NER.MEM import MEMM
from StartCodes.NER import MEM
import pickle
from django.shortcuts import render
from django.http import HttpResponse


def classify_sentence(request, self=None):
    if request.method == 'POST':
        sentence = request.POST['sentence']
        classifier = nltk.data.load('./StartCodes/model.pkl', format='pickle')
        prediction = predict_sentence(sentence, classifier, MEMM, self)
        return render(request, 'Output.html',
                      {'sentence': sentence, 'classification_result': prediction})
    else:
        return render(request, 'Input.html')
