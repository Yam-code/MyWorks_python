# ChatGPT(LangChain)

# ChatGPTに加えてLangChainを用いることで、段階的な処理を行いながら回答を生成するBotの作成を目指す。
# 今回は最新のAI関連ニュースについての応答を生成するBotを題材にする。

# 1. 決まった一つの処理を行うモデル
# まず、AI専門ニュースサイト (https://aismiley.co.jp/ai-news_category/generative-ai/) から単一の質問に回答するBotを作成する。
# 初めに、必要なライブラリをインストールする。

# %%
#!pip install openai langchain llama-index google-search-results faiss-cpu
#!pip install nbconvert jupyter_contrib_nbextensions jupyter_nbextensions_configurator notebook
#!pip install --upgrade notebook==6.4.12

# %%
import os
# 環境変数をセットする
os.environ["OPENAI_API_KEY"] = ""
os.environ["SERP_API_KEY"] = ""

SERP_API_KEY = os.environ["SERP_API_KEY"]

# %%
# まず、AI専門ニュースサイト (https://aismiley.co.jp/ai-news_category/generative-ai/) から文書を取得し、その文書を一定数のトークンの集まりのチャンクに分解する。


import re
from langchain.document_loaders import WebBaseLoader
# ニュースサイトからテキストを取得する
loader = WebBaseLoader(
"https://aismiley.co.jp/ai-news_category/generative-ai/"
)
documents= loader.load()
# 簡易的な前処理を行う
documents[0].page_content = re.sub(r"\n+", "\n", documents[0].page_content)

# %%
from langchain.text_splitter import RecursiveCharacterTextSplitter
# テキストスプリッターを初期化する
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
# テキストをチャンクに分割する
texts = text_splitter.split_documents(documents)

print(texts[0])

# %%
# 埋め込みを用いてクエリに最も関連するチャンクを抽出するRetrieverを作成する。


from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
# 埋め込みを初期化する
embeddings = OpenAIEmbeddings()
# ベクターストアに文書と埋め込みを格納
db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever()

# %%
# Retrieverを用いて、入力されたクエリに関連する該当部分を参照し回答させる。

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
# LLM ラッパーを読み込む
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
# チェーンを作り、それを使って質問に答える
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    output_key="description_str")

question= "最も新しいニュースは何ですか？"
print(qa_chain.run(question))

# %%
# 2. 決まった複数の処理を行うBot
# 次に、AI関連ニュースサイトから質問に回答し、その情報の詳細を要約するBotを作成する。
# まず、最新ニュース情報を入力として、それを詳細に要約するChainを作成する。

from langchain import LLMChain, PromptTemplate
day_qa_template_ja = """文脈情報は以下の通りです。

---------------------
{description_str}
---------------------
以前の知識を用いず、この文脈情報で以下の問いに答えてください。:\

文脈情報で言及されたものを詳しく要約してください。
"""
day_qa_prompt_template = PromptTemplate(
input_variables=["description_str"],
template=day_qa_template_ja,
)
day_qa_chain = LLMChain(llm=llm,
                        prompt=day_qa_prompt_template,
                        output_key="answer_str")

# %%
# その後、(1)で作成したサイト内で最も新しいニュースを答えるchainとその情報を要約させるchainを結合する。

from langchain.chains import SimpleSequentialChain
garbage_qa_chain = SimpleSequentialChain(chains=[qa_chain, day_qa_chain],

verbose=True)


# %%
garbage_qa_chain.run(question)

# %%
# 3. 自由に与えられたツールを使って回答するBot
# 最後に、インターネットを自由に検索し、最新のニュースとその要約を回答するBotを作成する。
# このためにAgentと呼ばれる機能を利用する。
# まず、インターネットを検索する機能を利用するため、SerpAPI https://serpapi.com/ のAPIキーを発行する。

# 次に、エージェントが利用するツールを定義する

from langchain import SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
# 検索するAPIのラッパーを用意する
search = SerpAPIWrapper(
serpapi_api_key=SERP_API_KEY,
params = {"engine": "bing","gl": "jp","hl": "ja"}
)

"""
list = []
def memorize():
    list.append("{summarizedText}")
    return
"""

# ツールにエージェントが利用する機能を格納する
tools = [
Tool(
name="Search",
func=search.run,
description="useful for when you need to answer questions what details about input text"
),
Tool(
    name="Summarize",
    func=day_qa_chain.run,
    description="useful for summarizing text you have searched for"
)
]

# %%
# 次にエージェントがどのように思考するかを示すプロンプトを作成する
# 備考:安定した推論を行うため、英語でのプロンプトにしている

# エージェントが利用するプロンプトを作成する
template = """Must Answer the following questions.You have access to the following tools:
{tools}
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer (in Japanese): the final answer to the original input question in Japanese
Question: {input}
{agent_scratchpad}"""

# %%
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import HumanMessage
from typing import List, Union
# エージェントのプロンプトのテンプレートを定義する
class CustomPromptTemplate(BaseChatPromptTemplate):
    template: str
    tools: List[Tool]
    def format_messages(self, kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(kwargs)
        return [HumanMessage(content=formatted)]
prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    input_variables=["input", "intermediate_steps"]
)



# %%
# エージェントの出力を追跡するパーサーを作成する

from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish, HumanMessage
# エージェントの出力のパーサーを作成する
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer (in Japanese):" in llm_output:
            return AgentFinish(
                return_values={
                "output": llm_output.split("Final Answer (in Japanese):")[-1].strip()
                },
            log=llm_output,
            )
        # 行動と行動のための入力をパースする
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # 行動と行動のための入力を返す
        return AgentAction(tool=action,
            tool_input=action_input.strip(" ").strip('"'),
            log=llm_output)

output_parser = CustomOutputParser()

# %%
# 以上のものを用いて、自由にツールを用いて与えられたタスクを解こうとするエージェントを設定する

from langchain.agents import LLMSingleActionAgent
from langchain.agents import AgentExecutor
# エージェントを設定する
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
llm_chain=llm_chain,
output_parser=output_parser,
stop=["\nObservation:"],
allowed_tools=tool_names
)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent,

tools=tools,
verbose=True)

# %%
agent_executor.run("最新のAI関連の技術革新について3つほど調べ、それぞれを要約してください。")

# %%
# わかったこと・わからなかったこと
#     わかったこと：生成AIは様々なサービスやAPIと連携してこそ真価を発揮するということ
# 
#     わからなかったこと：同じプロンプトでも出力ごとに大きな差ができており、それを抑えるためのプロンプトエンジニアリングが難しいと感じた。
