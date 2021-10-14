#!/usr/bin/env python3
"""
0x13. QA Bot
"""
question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """
    Answers questions from a reference text:

    Reference: the reference text
    If the answer cannot be found in the reference text, respond with Sorry,
    I do not understand your question.
    """
    while True:
        question = input("Q: ").lower()

        if question in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            exit()
        else:
            answer = question_answer(question, reference)
            if answer is None:
                print("A: Sorry, I do not understand your question.")
            else:
                print("A:", answer)
