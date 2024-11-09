API_KEY ="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"
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.prompts import (
    PromptTemplate,
     SystemMessagePromptTemplate,
     HumanMessagePromptTemplate,
     ChatPromptTemplate,
 )
from langchain_core.output_parsers import StrOutputParser 

review_template_str = """Your job is to use patient
 reviews to answer questions about their experience at a
 hospital. Use the following context to answer questions.
 Be as detailed as possible, but don't make up any information
 that's not from the context. If you don't know an answer, say
 you don't know.

 {context}
 """

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=review_template_str,
    )
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="{question}",
    )
)
messages = [review_system_prompt, review_human_prompt]

review_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=messages,
)

chat_model = HuggingFaceHub(repo_id = "Qwen/Qwen2.5-1.5B", huggingfacehub_api_token = API_KEY)

# will make the model’s response more readable.
output_parser = StrOutputParser()

# chain review_prompt_template and chat_model together.
review_chain = review_prompt_template | chat_model | output_parser