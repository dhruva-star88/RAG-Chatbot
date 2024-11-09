from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_intro.chatbot import chat_model


messages = [
     SystemMessage(
        content="""You're an assistant knowledgeable about
        healthcare. Only answer healthcare-related questions."""
    ),
    HumanMessage(content="give me info blood pressure"),
]

msg = chat_model.invoke(messages)
print(msg)