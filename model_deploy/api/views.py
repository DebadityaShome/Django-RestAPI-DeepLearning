from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.shortcuts import render
from scipy.special import softmax
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Create your views here.
@api_view(['GET'])
def index_page(request):
    return_data = {
        "error" : "0",
        "message" : "Successful",
    }
    return Response(return_data)

@api_view(['POST'])
def emotion(request):
    text=str(request.data.get('text',None))
  
    tokenizer = AutoTokenizer.from_pretrained("nateraw/bert-base-uncased-emotion")

    model = AutoModelForSequenceClassification.from_pretrained("nateraw/bert-base-uncased-emotion")


    labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    s = scores
    sadness = float(s[0])
    joy = float(s[1])
    love = float(s[2])
    anger = float(s[3])
    fear = float(s[4])
    surprise = float(s[5])
    result =  {
	'sadness': sadness,
	'joy': joy,
	'love': love,
	'anger': anger,
	'fear': fear,
	'surprise': surprise
    }

    return Response(result)

# Create your views here.
