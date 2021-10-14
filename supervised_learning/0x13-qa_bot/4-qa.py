#!/usr/bin/env python3
"""
0x13. QA Bot
"""
qa_0 = __import__('0-qa').question_answer
semantic_s = __import__('3-semantic_search').semantic_search


def question_answer(corpus_path):
    """
    Answers questions from multiple reference texts:
    corpus_path: to the corpus of reference documents
    """
    while True:
        question = input("Q: ").lower()

        if question in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            exit()
        else:
            text = semantic_s(corpus_path, question)
            answer = qa_0(question, text)
            if answer is None:
                print("A: Sorry, I do not understand your question.")
            else:
                print("A:", answer)
