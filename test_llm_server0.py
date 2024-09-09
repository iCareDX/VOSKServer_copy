# from langchain.schema import AIMessage, HumanMessage, SystemMessage
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import asyncio
import websockets
import json
import os

from langchain_openai import ChatOpenAI

API_KEY = "EMPTY"
API_BASE = "http://iCareDXnoMacBook-Air.local:8000/v1"
MODEL = "llama-3"

# from langchain.prompts import (
from langchain_core.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate,
)

from langchain_core.messages import SystemMessage
from langchain_core.messages import BaseMessage

from langchain_core.output_parsers import StrOutputParser

# from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from pydantic import Field
from typing import Sequence

# from langchain.globals import set_debug, set_verbose
# set_debug(True)
# set_verbose(True)

# import socket
# ipv4_addr = socket.gethostbyname('iCareDXnoMacBook-Air.local')
# API_BASE = f"http://{ipv4_addr}:8000/v1"

llm = ChatOpenAI(model_name=MODEL, 
                  openai_api_key=API_KEY, 
                  openai_api_base=API_BASE,
                  streaming=True, 
                #   callbacks=[StreamingStdOutCallbackHandler()] ,
                  temperature=0)

# Template setup
template = """
あなたは日本語で会話する高齢者介護スタッフです。あなたのプロフィールは以下です。
名前: わんこ
性別: 男
年齢: 25歳
出身: 東京都調布市
職業: 介護スタッフ（5年目）、公認介護度認定士
スキル: 長谷川式認知症診断法',
趣味: ハイキング、映画鑑賞、楽器演奏（ドラム）。
あなたは、高齢者のお世話が仕事です。特に、高齢者の悩みに対して身の上相談をして助けてあげたいと思っています。
高齢者の老化に伴う肉体的、精神的な痛みに対して、相談に乗ってあげてください。
高齢者と会話する時には、簡潔にわかりやすく答えてください。
わからない質問には、適当に答えないで、素直にわかりませんと答えてください。
自分のプロフィールについては聞かれた時だけに答えてください。また、必要なら下記のコンテクスト情報を参考にして回答してください。
"""

# conversational_memory_length = 5

# memory = ConversationBufferWindowMemory(
#     k=conversational_memory_length, memory_key="history", return_messages=True
# )

# Chat prompt template setup
# prompt = ChatPromptTemplate.from_messages([
#     SystemMessagePromptTemplate.from_template(template),
#     MessagesPlaceholder(variable_name="history"),
#     HumanMessagePromptTemplate.from_template("{input}")
# ])

# from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferMemory

# TOKEN_LIMIT = 2048

# class TruncatedConversationBufferMemory(ConversationBufferMemory):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def get_token_count(self, text):
#         # Simple token count estimation
#         return len(text.split())

#     def truncate_history(self):
#         total_tokens = sum(self.get_token_count(msg.content) for msg in self.messages)
#         while total_tokens > TOKEN_LIMIT:
#             removed_message = self.messages.pop(0)
#             total_tokens -= self.get_token_count(removed_message.content)

#     def add_message(self, message):
#         super().add_message(message)
#         self.truncate_history()

# Memory setup with truncation
# memory = TruncatedConversationBufferMemory(return_messages=True)

# Conversation chain setup
# conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)


class LimitedChatMessageHistory(ChatMessageHistory):
    max_messages: int = Field(default=10)

    def __init__(self, max_messages=10):
        super().__init__()
        self.max_messages = max_messages

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        super().add_messages(messages)
        self._limit_messages()

    def _limit_messages(self):
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

store = {}
memory = LimitedChatMessageHistory(max_messages=5)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = memory
    return store[session_id]

def llm_main(user_input):
    if user_input:
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=template),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{user_input}"),
            ]
        )

        parser = StrOutputParser()

        chain = prompt | llm | parser

        runnable_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="user_input",
            history_messages_key="history",
        )
        response = runnable_with_history.invoke(
            {"user_input": user_input},
            config={"configurable": {"session_id": "123"}},
        )

        # print(type(response))
        # print(response)
        return response
       
def main_():
    while True:
        user_input = input("Please input: ")
        if user_input.lower() == "exit":
            print("Goodbye")
            break
 
            response = llm_main(user_input=user_input)
 
            print("User: ", user_input)
            print("Assistant: ",response)
    

async def handle_connection(websocket, path):
    async for message in websocket:
        print(f"Received message: {message}")

        # response = conversation.predict(input=message)
        response = llm_main(user_input=message)

# stream output
        # prompt = ChatPromptTemplate.from_messages(
        #     [
        #         SystemMessage(content=template),
        #         MessagesPlaceholder(variable_name="history"),
        #         HumanMessagePromptTemplate.from_template("{user_input}"),
        #     ]
        # )

        # parser = StrOutputParser()

        # chain = prompt | llm | parser

        # runnable_with_history = RunnableWithMessageHistory(
        #     chain,
        #     get_session_history,
        #     input_messages_key="user_input",
        #     history_messages_key="history",
        # )
    
        # async for chunk in runnable_with_history.astream({"user_input": message},
        #     config={"configurable": {"session_id": "123"}},
        # ):
        #     await websocket.send(chunk)
        #     print(f"Sent response: {chunk}")
# stream output
 
        await websocket.send(response)
        print(f"Sent response: {response}")

async def main():
    async with websockets.serve(handle_connection, "localhost", 8765):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())

