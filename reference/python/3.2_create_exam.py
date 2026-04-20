import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_gigachat.chat_models import GigaChat
from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings

###################################################
###   ИНИЦИАЛИЗАЦИЯ ПЕРЕМЕННЫХ СРЕДЫ И МОДЕЛИ   ###
###################################################

# Загружаем переменные окружения из файла .env
load_dotenv()

# Строка подключения к PostgreSQL. Все параметры берутся из переменных окружения.
CONNECTION_STRING = (
    f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)

# Имя коллекции (таблицы) в векторной базе данных
COLLECTION_NAME = "exam_embeddings"


###############################################################
###   ЗАГРУЗКА И ИНДЕКСАЦИЯ ДОКУМЕНТОВ ИЗ ЛОКАЛЬНЫХ ФАЙЛОВ  ###
###############################################################


def process_documents():
    """
    Читает все Markdown-файлы из директории ./exams,
    разбивает их на чанки и сохраняет эмбеддинги в PostgreSQL.
    Вызывается один раз при первом запуске.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        length_function=len,
    )

    documents = []
    exams_dir = Path("./exams")

    # Обходим все .md-файлы в директории с экзаменами
    for exam_file in exams_dir.glob("*.md"):
        try:
            # encoding="utf-8" задаём явно, чтобы корректно читать кириллицу на любой ОС
            loader = TextLoader(str(exam_file), encoding="utf-8")
            doc = loader.load()

            # Разбиваем документ на небольшие перекрывающиеся чанки
            split_docs = text_splitter.split_documents(doc)

            # Записываем имя файла в метаданные каждого чанка для отслеживания источника
            for split_doc in split_docs:
                split_doc.metadata["source"] = exam_file.name

            documents.extend(split_docs)
            print(f"Обработан файл: {exam_file.name}")

        except Exception as e:
            print(f"Ошибка при обработке файла {exam_file}: {e}")

    # Создаём эмбеддинги — числовые векторы для каждого чанка
    embeddings = GigaChatEmbeddings(
        credentials=os.getenv("GIGACHAT_CREDENTIALS"), verify_ssl_certs=False
    )

    # Создаём векторную базу данных в PostgreSQL и записываем все эмбеддинги
    db = PGVector.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
    )

    print(f"Загружено {len(documents)} фрагментов в базу данных")
    return db


###############################################################
###   ЗАПРОС К ВЕКТОРНОЙ БАЗЕ И ГЕНЕРАЦИЯ НОВОГО ЭКЗАМЕНА   ###
###############################################################


def query_vector_db(query: str):
    """
    Выполняет RAG-запрос: ищет похожие экзаменационные задания
    и генерирует новое задание на основе найденного контекста.
    """
    # Подключаем эмбеддинги для преобразования запроса в вектор
    embeddings = GigaChatEmbeddings(
        credentials=os.getenv("GIGACHAT_CREDENTIALS"), verify_ssl_certs=False
    )

    # Подключаемся к уже существующей векторной базе (без повторного наполнения)
    db = PGVector(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
    )

    # retriever ищет k=10 наиболее похожих чанков по запросу
    retriever = RunnableLambda(db.similarity_search).bind(k=10)

    # Создаём языковую модель для генерации итогового ответа
    model = GigaChat(
        credentials=os.getenv("GIGACHAT_CREDENTIALS"),
        verify_ssl_certs=False,
    )

    # Промпт для методиста: инструктирует модель создать новое задание
    # по образцу контекста из предыдущих экзаменов
    message = """
    Ты - опытный методист, который создает экзаменационные задания для студентов программирования. 
    Тебе нужно создать новое экзаменационное задание, опираясь на формат и структуру предыдущих экзаменов.

    Контекст из предыдущих экзаменов:
    {context}

    Создай новое экзаменационное задание по следующему запросу:
    {question}

    Твой ответ должен быть в формате markdown и обязательно содержать:
    - Название проекта
    - Краткое описание проекта
    - Релизы выполнения проекта, начиная с 0
    - В каждом релизе описана какая-то бизнес-фича, которая должна быть реализована студентами
    - Описание в релизе должно быть достаточно абстрактным, но понятным для студентов
    - В проекте должна быть регистрация и авторизация
    - В релизах должны быть описаны все CRUD операции
    - Количество таблиц в БД должно быть ровно 2
    """

    prompt = ChatPromptTemplate.from_messages([("human", message)])

    # Собираем RAG-цепочку: retriever подставляет контекст, а модель генерирует ответ
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} | prompt | model
    )
    response = rag_chain.invoke(query)

    return response.content


###############################################################
###   ТОЧКА ВХОДА                                           ###
###############################################################

if __name__ == "__main__":
    # Первый запуск — загружаем файлы из ./exams и наполняем базу данных.
    # При повторных запусках эту строку можно закомментировать.
    db = process_documents()
    print("База данных успешно создана")

    # Генерируем новое экзаменационное задание
    result = query_vector_db("создай ТЗ для экзамена: магазин лыж и сноубордов")
    print("\nРезультат запроса:")
    print(result)

    # Сохраняем сгенерированное задание в markdown-файл
    with open("assessment-gigachat-new.md", "w", encoding="utf-8") as f:
        f.write(result)
    print("\nРезультат сохранён в файл assessment-gigachat-new.md")
