import markdown
from django.shortcuts import render
from django.utils.html import escape

from rag.generation import answer_question


def ask(request):
    answer_html = None
    sources = []
    question = ""
    error = None

    if request.method == "POST":
        question = request.POST.get("question", "").strip()
        if question:
            try:
                answer, sources = answer_question(question)
                # Escape first: the LLM's raw text shouldn't be interpreted as
                # HTML, only the Markdown syntax it was asked to reply in
                # (bold, lists, etc.) should be rendered.
                answer_html = markdown.markdown(escape(answer), extensions=["nl2br"])
            except Exception as e:
                error = str(e)

    return render(
        request,
        "rag/index.html",
        {"answer_html": answer_html, "sources": sources, "question": question, "error": error},
    )
