from django.shortcuts import render

from rag.generation import answer_question


def ask(request):
    answer = None
    sources = []
    question = ""
    error = None

    if request.method == "POST":
        question = request.POST.get("question", "").strip()
        if question:
            try:
                answer, sources = answer_question(question)
            except Exception as e:
                error = str(e)

    return render(
        request,
        "rag/index.html",
        {"answer": answer, "sources": sources, "question": question, "error": error},
    )
