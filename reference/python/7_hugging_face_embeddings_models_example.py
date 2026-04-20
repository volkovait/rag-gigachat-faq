import torch
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_gigachat.chat_models import GigaChat
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain.chains import RetrievalQA

##########################################
###   ИНИЦИАЛИЗАЦИЯ ПЕРЕМЕННЫХ СРЕДЫ   ###
##########################################

# Получаем ключ API GigaChat из переменных окружения
GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS")

##########################################################
###   ИНИЦИАЛИЗАЦИЯ МОДЕЛИ ЭМБЕДДИНГОВ HUGGING FACE    ###
##########################################################

# Загружаем локальную модель Giga-Embeddings с HuggingFace.
# Модель запускается на GPU (если доступен), иначе на CPU.
# trust_remote_code=True требуется, так как модель использует кастомный код архитектуры.
# torch_dtype=bfloat16 снижает потребление памяти при сохранении точности.
embeddings = HuggingFaceEmbeddings(
    model_name='ai-sage/Giga-Embeddings-instruct',
    model_kwargs={
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'trust_remote_code': True,
        'model_kwargs': {'torch_dtype': torch.bfloat16},
    },
    # Instruct-промпт передаётся как параметр верхнего уровня HuggingFaceEmbeddings,
    # а не внутри model_kwargs — направляет модель на поиск релевантных отрывков по вопросу,
    # что повышает точность семантического поиска по сравнению с базовым режимом
    query_instruction='Instruct: Given a question, retrieve passages that answer the question\nQuery: ',
    encode_kwargs={
        # Нормализация приводит векторы к единичной длине,
        # что позволяет использовать косинусное сходство как метрику близости
        'normalize_embeddings': True
    }
)

#############################################################
###   ПОДГОТОВКА БАЗЫ ЗНАНИЙ (КОРПУС ДОКУМЕНТОВ)          ###
#############################################################

# Создаём небольшую базу знаний из трёх документов о России.
# В реальном сценарии документы загружаются из файлов, баз данных или веб-источников.
documents = [
    Document(page_content="Москва — столица России и крупнейший город страны. Население составляет более 12 миллионов человек."),
    Document(page_content="Санкт-Петербург — второй по величине город России, культурная столица. Основан в 1703 году Петром I."),
    Document(page_content="Русский язык относится к восточнославянской группе языков. Является официальным языком России."),
]

###############################################
###   СОЗДАНИЕ ВЕКТОРНОГО ХРАНИЛИЩА         ###
###############################################

# Преобразуем документы в векторы эмбеддингов и сохраняем в оперативной памяти.
# InMemoryVectorStore — простое хранилище без внешних зависимостей,
# подходит для прототипирования и небольших баз знаний.
vector_store = InMemoryVectorStore.from_documents(
    documents=documents,
    embedding=embeddings
)

#############################################################
###   ИНИЦИАЛИЗАЦИЯ ЯЗЫКОВОЙ МОДЕЛИ GIGACHAT (LLM)        ###
#############################################################

# Создаём экземпляр GigaChat для генерации финальных ответов на основе найденных фрагментов.
# temperature=0.1 делает ответы детерминированными и фактически точными (минимум случайности).
gigachat = GigaChat(
    credentials=GIGACHAT_CREDENTIALS,  # Получите в личном кабинете разработчика Сбера
    verify_ssl_certs=False,
    model="GigaChat-Pro",  # GigaChat-Lite — более быстрая и дешёвая альтернатива
    temperature=0.1,
    max_tokens=1024
)

##############################################################
###   СБОРКА RAG-ЦЕПИ (RETRIEVAL-AUGMENTED GENERATION)    ###
##############################################################

# Объединяем ретривер и LLM в единую цепь RetrievalQA.
# chain_type="stuff" — простейшая стратегия: все найденные чанки вставляются в один промпт.
# Подходит, когда суммарный объём найденных фрагментов не превышает контекстное окно модели.
# search_kwargs={"k": 2} — возвращаем 2 наиболее близких по смыслу документа к запросу.
qa_chain = RetrievalQA.from_chain_type(
    llm=gigachat,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,  # Включаем возврат источников для проверки релевантности
    verbose=True
)

##############################################################
###   ТЕСТИРОВАНИЕ ЦЕПИ: ВОПРОС И ВЫВОД ОТВЕТА С RAG      ###
##############################################################

# Задаём вопрос и запускаем полный RAG-пайплайн:
# 1) ретривер находит 2 релевантных чанка по семантическому сходству
# 2) LLM генерирует ответ, опираясь только на найденные фрагменты
query = "Какие крупные города есть в России?"
result = qa_chain.invoke({"query": query})

print(f"Вопрос: {query}")
print(f"Ответ: {result['result']}")
print("\nИспользованные источники:")
for doc in result['source_documents']:
    print(f"- {doc.page_content}")
