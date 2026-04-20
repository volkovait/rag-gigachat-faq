import os

from chromadb.config import Settings
from dotenv import load_dotenv
from langchain import hub
from langchain.schema import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_gigachat.chat_models import GigaChat
from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings

###################################################
###   ИНИЦИАЛИЗАЦИЯ ПЕРЕМЕННЫХ СРЕДЫ И МОДЕЛИ   ###
###################################################

# Загружаем переменные окружения из файла .env
load_dotenv()

# Получаем ключ API GigaChat из переменных окружения
GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS")

# Создаём экземпляр языковой модели
llm = GigaChat(verify_ssl_certs=False, credentials=GIGACHAT_CREDENTIALS)

###########################################################
###   ТЕСТИРОВАНИЕ ЗАПРОСА С НЕДОСТАТОЧНЫМИ ЗНАНИЯМИ    ###
###########################################################

# Задаём вопрос модели без дополнительного контекста.
# Ожидаем неточный или устаревший ответ — это исходная точка для сравнения с RAG.
question = "Какая планета в солнечной системе имеет больше всего спутников?"
response = llm.invoke([HumanMessage(content=question)])
response_content = response.content
print("Ответ без RAG:", response_content)

############################################################
####   ИНИЦИАЛИЗАЦИЯ БАЗЫ ДАННЫХ И ВСТАВКА ДОКУМЕНТОВ    ###
############################################################

# Загружаем исходный текстовый файл с информацией о солнечной системе
loader = TextLoader("./data/solar_ru.txt", encoding="utf-8")
documents = loader.load()

# Разбиваем текст на перекрывающиеся чанки по 1000 символов.
# chunk_overlap=200 обеспечивает плавный переход между чанками —
# контекст с границы одного чанка попадает в начало следующего.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
documents = text_splitter.split_documents(documents)

# Создаём эмбеддинги — числовые векторы, представляющие смысл каждого чанка.
# Семантически похожие тексты будут иметь близкие векторы.
embeddings = GigaChatEmbeddings(
    credentials=GIGACHAT_CREDENTIALS, verify_ssl_certs=False
)

# Создаём векторную базу данных Chroma и индексируем все чанки.
# anonymized_telemetry=False отключает отправку анонимной статистики.
db = Chroma.from_documents(
    documents,
    embeddings,
    client_settings=Settings(anonymized_telemetry=False),
)

################################################
#####   ПОВТОРНЫЙ ЗАПРОС С ПРИМЕНЕНИЕМ RAG   ###
################################################

# Ищем k=4 наиболее релевантных чанка по семантическому сходству с вопросом
retrieved_docs = db.similarity_search(question, k=4)

# Загружаем готовый RAG-промпт из LangChain Hub.
# Промпт инструктирует модель отвечать только на основе переданного контекста.
prompt = hub.pull("rlm/rag-prompt")

# Объединяем найденные чанки в единый текст-контекст
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

# Подставляем вопрос и контекст в шаблон промпта и получаем ответ
messages = prompt.invoke({"question": question, "context": docs_content})
new_response = llm.invoke(messages)
print("Ответ с RAG:", new_response.content)
