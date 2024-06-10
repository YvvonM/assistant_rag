from django.shortcuts import render
from django.http import JsonResponse
from qarag.parentchild import get_answer_from_rag_chain
# Create your views here.

def qa_rag(request):


    question = "Recommend me roles in software engineering"
    chat_history = []

    answer = get_answer_from_rag_chain(question, [])

    chat_history.push(question)

    return JsonResponse({'answer': answer, "chat_history": chat_history})
