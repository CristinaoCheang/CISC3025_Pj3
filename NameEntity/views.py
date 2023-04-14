from StartCodes.NER.MEM import predict_sentence
from django.http import JsonResponse
from django.shortcuts import render
from StartCodes.NER.MEM import MEMM
from StartCodes.NER import MEM


def predict(text, model):
    prediction = MEM.predict_sentence(text, model)
    return prediction

def home(request):
    if request.method == 'POST':
        # 获取POST请求中的数据
        text = request.POST['text']

        # 对数据进行预测
        result = predict(text, model='model.pkl')

        # 返回预测结果
        return JsonResponse({'result': result})
    else:
        # 显示HTML页面
        return render(request, 'ner.html')