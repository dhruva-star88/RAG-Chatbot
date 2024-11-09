from langchain_intro.chatbot import review_chain

context = "I had a great stay!"
question = "Did anyone have a positive experience?"

res = review_chain.invoke({"context": context, "question": question})

print(res)