import os
import os.path

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

#################################################
###    ИНИЦИАЛИЗАЦИЯ ОКРУЖЕНИЯ И КРЕДЕНШИАЛОВ ###
#################################################

# Загружаем переменные окружения из файла .env
load_dotenv()

# Область доступа: только чтение файлов на Google Drive.
# Если изменить SCOPES — нужно удалить token.json, чтобы пройти авторизацию заново.
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

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
        # Открываем браузер для авторизации через Google OAuth2.
        # credentials.json скачивается из Google Cloud Console.
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)

    # Сохраняем новый токен для следующего запуска
    with open("token.json", "w") as token:
        token.write(creds.to_json())

# Создаём клиент Google Drive API
service = build("drive", "v3", credentials=creds)

#################################################
###    ПОЛУЧЕНИЕ СПИСКА ФАЙЛОВ С GOOGLE DRIVE ###
#################################################

# Запрашиваем список до 10 файлов. fields определяет, какие поля вернуть.
# nextPageToken нужен для постраничной навигации при большом количестве файлов.
results = (
    service.files()
    .list(pageSize=10, fields="nextPageToken, files(id, name)")
    .execute()
)
items = results.get("files", [])

if not items:
    print("Файлы не найдены.")
else:
    print("Файлы на Google Drive:")
    for item in items:
        print(f"  {item['name']} ({item['id']})")
