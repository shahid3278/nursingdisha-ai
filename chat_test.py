from rag import ask_rag

while True:

    q = input("Ask: ")

    print(
        ask_rag(q)
    )