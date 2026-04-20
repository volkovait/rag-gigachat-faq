import base64
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from gigachat.exceptions import ResponseError
from github import Github
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_gigachat.chat_models import GigaChat
from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

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
COLLECTION_NAME = "assessment_embeddings"

###############################################################
###   ЗАГРУЗКА README-ФАЙЛОВ ЭКЗАМЕНОВ ИЗ GITHUB            ###
###############################################################


def fetch_assessment_readmes(github_token: str) -> list[Document]:
    """
    Загружает README.md файлы из репозиториев организации Elbrus-Bootcamp,
    чьи имена начинаются с 'assessment-2'. Каждый README разбивается
    на отдельные документы по секциям Release.

    Возвращает список объектов Document для дальнейшей индексации.
    """
    g = Github(github_token)
    org = g.get_organization("Elbrus-Bootcamp")
    documents = []

    # Создаём директорию exams для сохранения скачанных файлов локально
    exams_dir = Path("./exams")
    exams_dir.mkdir(exist_ok=True)

    for repo in org.get_repos():
        # Фильтруем только репозитории с экзаменами (учитываем опечатки в именах)
        if not (
            repo.name.startswith("assessment-2")
            or repo.name.startswith("asessment-2")
            or repo.name.startswith("assesment-2")
            or repo.name.startswith("asesment-2")
        ):
            continue

        try:
            readme = None
            # Пробуем несколько вариантов написания имени файла
            for filename in ("README.md", "Readme.md", "readme.md"):
                try:
                    readme = repo.get_contents(filename, ref=repo.default_branch)
                    break
                except Exception:
                    continue

            if readme is None:
                continue

            # Содержимое файла в GitHub хранится в base64 — декодируем его
            content = base64.b64decode(readme.content).decode("utf-8")

            # Сохраняем копию файла локально для последующего использования
            exam_file = exams_dir / f"{repo.name}.md"
            exam_file.write_text(content, encoding="utf-8")

            lines = content.split("\n")

            # Извлекаем заголовок — первая непустая строка без символов `#`
            title = next(
                (line.strip("# ") for line in lines if line.strip()), "Untitled"
            )

            # Переменные для сборки текущего релиза
            current_release = []
            release_number = None

            for line in lines:
                # Ищем заголовки вида "## Release 0" или "### Release 1.1"
                if "## Release" in line or "### Release" in line:
                    # Сохраняем накопленный предыдущий релиз перед началом нового
                    if current_release and release_number is not None:
                        release_content = "\n".join(current_release)
                        documents.append(
                            Document(
                                page_content=release_content,
                                metadata={
                                    "source": f"github.com/Elbrus-Bootcamp/{repo.name}",
                                    "exam_title": title,
                                    "release": release_number,
                                },
                            )
                        )

                    # Извлекаем номер/название релиза — всё, что стоит после слова "Release"
                    release_number = line.split("Release")[-1].strip()
                    current_release = [line]
                elif release_number is not None:
                    # Накапливаем строки текущего релиза
                    current_release.append(line)

            # Добавляем последний релиз, который не был закрыт новым заголовком
            if current_release and release_number is not None:
                release_content = "\n".join(current_release)
                documents.append(
                    Document(
                        page_content=release_content,
                        metadata={
                            "source": f"github.com/Elbrus-Bootcamp/{repo.name}",
                            "exam_title": title,
                            "release": release_number,
                        },
                    )
                )

            # Дополнительно сохраняем весь README как единый документ
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": f"github.com/Elbrus-Bootcamp/{repo.name}",
                        "exam_title": title,
                        "release": "full",
                    },
                )
            )

        except Exception as e:
            print(f"Ошибка при обработке {repo.name}: {e}")

    return documents


###############################################################
###   КЛАСС С ОГРАНИЧЕНИЕМ СКОРОСТИ ЗАПРОСОВ К API          ###
###############################################################


class RateLimitedGigaChatEmbeddings(GigaChatEmbeddings):
    """
    Наследует GigaChatEmbeddings и добавляет автоматические повторные попытки
    при ошибках превышения лимита запросов (rate limit).
    wait_exponential: пауза между попытками увеличивается экспоненциально (4–10 сек).
    stop_after_attempt(3): максимум 3 попытки перед тем, как выбросить исключение.
    """

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(ResponseError),
    )
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Задержка перед каждым батчем, чтобы не превышать лимит API
        time.sleep(1)
        return super().embed_documents(texts)

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(ResponseError),
    )
    def embed_query(self, text: str) -> list[float]:
        # Задержка перед каждым одиночным запросом
        time.sleep(1)
        return super().embed_query(text)


###############################################################
###   ИНИЦИАЛИЗАЦИЯ ВЕКТОРНОЙ БАЗЫ ДАННЫХ                   ###
###############################################################


def initialize_vector_db():
    """
    Загружает документы с GitHub и наполняет векторную базу данных.
    Вызывается один раз при первом запуске.
    """
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise ValueError("Переменная окружения GITHUB_TOKEN не задана")

    # Загружаем документы из GitHub
    documents = fetch_assessment_readmes(github_token)

    # Разбиваем документы на небольшие чанки.
    # chunk_size=400 выбран с учётом лимитов токенов GigaChat на один запрос.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        length_function=len,
    )
    split_documents = text_splitter.split_documents(documents)

    embeddings = RateLimitedGigaChatEmbeddings(
        credentials=os.getenv("GIGACHAT_CREDENTIALS"),
        scope="GIGACHAT_API_PERS",
        verify_ssl_certs=False,
    )

    # Создаём векторное хранилище в PostgreSQL и записываем все эмбеддинги
    db = PGVector.from_documents(
        documents=split_documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
    )

    return db


###############################################################
###   ЗАПРОС К ВЕКТОРНОЙ БАЗЕ И ГЕНЕРАЦИЯ НОВОГО ЭКЗАМЕНА   ###
###############################################################


def query_vector_db(query: str):
    """
    Выполняет RAG-запрос: ищет похожие экзаменационные задания
    и генерирует новое задание на основе найденного контекста.
    """
    embeddings = RateLimitedGigaChatEmbeddings(
        credentials=os.getenv("GIGACHAT_CREDENTIALS"),
        scope="GIGACHAT_API_PERS",
        verify_ssl_certs=False,
    )

    # Подключаемся к уже существующей векторной базе (без повторного наполнения)
    db = PGVector(
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        embeddings=embeddings,
    )

    # retriever ищет k=10 наиболее похожих чанков по запросу
    retriever = RunnableLambda(db.similarity_search).bind(k=10)

    # GigaChat-Pro — более мощная версия модели для сложных задач генерации
    model = GigaChat(
        credentials=os.getenv("GIGACHAT_CREDENTIALS"),
        scope="GIGACHAT_API_PERS",
        model="GigaChat-Pro",
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

    Важно: задание должно быть уникальным, но следовать общей структуре и сложности предыдущих экзаменов.
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
    # Первый запуск — наполняем базу данных документами с GitHub.
    # При повторных запусках эту строку можно закомментировать.
    db = initialize_vector_db()

    # Генерируем новое экзаменационное задание по описанию предметной области
    result = query_vector_db(
        "Создай экзаменационное задание для приложения продажа домашних растений. "
        "Пользователи могут регистрироваться и авторизовываться, а после авторизации "
        "добавлять свои домашние растения, редактировать и удалять. "
        "Другие пользователи могут их просматривать."
    )
    print(result)
