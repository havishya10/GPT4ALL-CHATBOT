import os
import gradio as gr
from gpt4all import GPT4All
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler



template = """You are a helpful assistant to answer all user queries.
{chat_history}
User: {user_message}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "user_message"], template=template
)

memory = ConversationBufferMemory(memory_key="chat_history")

model = GPT4All("ggml-gpt4all-l13b-snoozy.bin")

from langchain.llms import GPT4All

# Callbacks support token-wise streaming
callbacks = [StreamingStdOutCallbackHandler()]

# Verbose is required to pass to the callback manager
llm = GPT4All(model="ggml-gpt4all-l13b-snoozy.bin", callbacks=callbacks, verbose=True)

# If you want to use a custom model add the backend parameter
# Check https://docs.gpt4all.io/gpt4all_python.html for supported backends
llm = GPT4All(model="ggml-gpt4all-l13b-snoozy.bin", backend="gptj", callbacks=callbacks, verbose=True)

llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True, memory=memory,)

def get_text_response(user_message,history):
    response = llm_chain.predict(user_message = user_message)
    return response

demo = gr.ChatInterface(get_text_response)

if __name__ == "__main__":
    demo.launch() #To create a public link, set `share=True` in `launch()`. To enable errors and logs, set `debug=True` in `launch()`.
