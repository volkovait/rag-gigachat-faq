import io
import os
import os.path

from chromadb.config import Settings
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_gigachat.chat_models import GigaChat
from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings

#################################################
###    ИНИЦИАЛИЗАЦИЯ ОКРУЖЕНИЯ И КРЕДЕНШИАЛОВ ###
#################################################

# Загружаем переменные окружения из файла .env
load_dotenv()

# Область доступа: только чтение файлов на Google Drive.
# Если изменить SCOPES — нужно удалить token.json, чтобы пройти авторизацию заново.
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS")

creds = None

# token.json создаётся автоматически после первой успешной авторизации.
# Он хранит токены доступа и обновления, чтобы не входить снова при каждом запуске.
if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)

# Если токена нет или он просрочен — запускаем процедуру авторизации
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        # Токен просрочен, но есть refresh_token — обновляем без переавторизации
        creds.refresh(Request())
    else:
        # Открываем браузер для авторизации через Google OAuth2
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)

    # Сохраняем новый токен для следующего запуска
    with open("token.json", "w") as token:
        token.write(creds.to_json())

# Создаём клиент Google Drive API
service = build("drive", "v3", credentials=creds)

####################################################
###    ПОИСК ФАЙЛА И ЗАГРУЗКА В ТЕКСТОВЫЙ ФОРМАТ ###
####################################################

# Поиск файла по точному названию. Можно изменить имя для поиска другого документа.
# query = "name = 'План на 1й день Мск'"
query = "name = 'Копия 2025 ШАБЛОН внутренней рассрочки - сделать копию!-copy'"
results = (
    service.files()
    .list(q=query, pageSize=10, fields="nextPageToken, files(id, name)")
    .execute()
)
items = results.get("files", [])

if not items:
    print("Файл не найден.")
    # Прекращаем выполнение — дальнейший код не имеет смысла без файла
    raise SystemExit("Выполнение прервано: файл не найден на Google Drive.")
else:
    print("Найденные файлы:")
    for item in items:
        print(f"  {item['name']} ({item['id']})")

# Берём первый найденный файл (предполагаем, что имя уникально)
file_id = items[0]["id"]

# Экспортируем Google Документ в текстовый формат (plain text).
# Google Docs нельзя скачать напрямую — только через export.
request = service.files().export_media(
    fileId=file_id,
    mimeType="text/plain",
)

# Скачиваем файл частями в оперативную память
fh = io.BytesIO()
downloader = MediaIoBaseDownload(fh, request)
done = False
while not done:
    status, done = downloader.next_chunk()
    print(f"Загрузка: {int(status.progress() * 100)}%")

document_name = "./data/dogovor.txt"

# Сохраняем скачанный контент в файл
with open(document_name, "wb") as f:
    f.write(fh.getvalue())

print("Файл успешно скачан в текстовом формате!")

##########################################################
###    РАЗБИЕНИЕ ФАЙЛА НА ЧАСТИ И СОЗДАНИЕ ЭМБЕДДИНГОВ ###
##########################################################

# Загружаем сохранённый текстовый файл
loader = TextLoader(document_name, encoding="utf-8")
documents = loader.load()

# Разбиваем текст на перекрывающиеся чанки по 500 символов
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)
documents = text_splitter.split_documents(documents)
print(f"Документ разбит на {len(documents)} частей")

# Создаём эмбеддинги — числовые векторы, представляющие смысл каждого чанка
embeddings = GigaChatEmbeddings(
    credentials=os.getenv("GIGACHAT_CREDENTIALS"), verify_ssl_certs=False
)

# Сохраняем чанки и их эмбеддинги в векторную базу данных Chroma
db = Chroma.from_documents(
    documents,
    embeddings,
    client_settings=Settings(anonymized_telemetry=False),
)
print("Эмбеддинги успешно созданы и сохранены в базу данных")

###########################################################
###    ПОИСК В ВЕКТОРНОЙ БАЗЕ ДАННЫХ И ВЫВОД РЕЗУЛЬТАТА ###
###########################################################

llm = GigaChat(
    credentials=GIGACHAT_CREDENTIALS,
    verify_ssl_certs=False,
)

# RetrievalQA — готовая цепочка LangChain, которая автоматически:
# 1. Ищет релевантные чанки в векторной базе
# 2. Передаёт их как контекст в языковую модель
# 3. Возвращает ответ модели
qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

# print(qa_chain.invoke("У кого взять корпоративную почту?")["result"])
# print(qa_chain.invoke("У кого заказать пропуск?")["result"])
print(
    qa_chain.invoke(
        "В течение какого времени предоставляется доступ к учебным материалам?"
    )["result"]
)
