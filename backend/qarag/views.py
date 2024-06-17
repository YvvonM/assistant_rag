from django.shortcuts import render
from django.http import JsonResponse
from qarag.parentchild import get_answer_from_rag_chain
import json

# # Create your views here.
# def qa_form(request):
#     return render(request,'/qarag/index.html')

def qa_rag(request):
    return render(request,'/qarag/index.html')
    if request.method == "POST":
        question = request.POST.get("question")
        
        print({question})
        
        chat_history = []

        (answer, _chat_history) = get_answer_from_rag_chain(question, [])

        chat_history.append(question)

        return JsonResponse({'answer': answer, "chat_history": chat_history})

