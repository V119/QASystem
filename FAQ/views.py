from django.shortcuts import render
from FAQ.FAQsystem import get_answer_list


# Create your views here.
def index(request):
    question = request.POST.get("question", None)
    if not question:
        return render(request, 'index.html', {})

    answers = get_answer_list(question)
    return render(request, 'index.html', {'answers': answers})
