# RAG (Retrieval-Augmented Generation) — практические примеры

Набор скриптов, демонстрирующих подход RAG (Retrieval-Augmented Generation) с использованием различных источников данных и векторных баз данных.

## Что такое RAG?

RAG — архитектурный подход, при котором языковая модель перед генерацией ответа получает релевантные фрагменты из базы знаний. Это позволяет давать точные ответы на вопросы о конкретных документах, которые модель «не видела» при обучении.

**Схема работы:**
```
Вопрос → Поиск похожих чанков → Контекст + Вопрос → Модель → Ответ
```

## Структура проекта

```
python/
├── 1_solar.py                    # Базовый RAG: данные о солнечной системе (Chroma)
├── 2_bulgakov.py                 # Базовый RAG: текст романа Булгакова (Chroma)
├── 3.1_load_from_github.py       # RAG с загрузкой README из GitHub (PostgreSQL + PGVector)
├── 3.2_create_exam.py            # RAG для генерации экзаменов из локальных файлов (PostgreSQL + PGVector)
├── 4_extract_from_google_docs.py # RAG с загрузкой документа из Google Docs (Chroma)
├── 5_web_cache.py                # RAG с PDF-документом (Chroma)
├── 6_get_files_from_drive.py     # Утилита: список файлов на Google Drive
├── data/                         # Исходные данные
│   ├── solar_ru.txt
│   ├── bulgakov_utf8.txt
│   ├── dogovor.txt
│   ├── first_day.txt
│   └── gotta-cache-em-all.pdf
├── exams/                        # Экзаменационные задания (создаётся автоматически)
└── requirements.txt
```

## Описание скриптов

### 1. `1_solar.py` — Базовый RAG с текстом о солнечной системе

- **Источник данных:** `data/solar_ru.txt`
- **Векторная БД:** Chroma (in-memory)
- **Что демонстрирует:**
  - Разницу между ответами модели без контекста и с RAG
  - Разбиение текста на чанки с перекрытием
  - Семантический поиск через `similarity_search`
  - Использование готового RAG-промпта из LangChain Hub (`rlm/rag-prompt`)

---

### 2. `2_bulgakov.py` — Базовый RAG с литературным текстом

- **Источник данных:** `data/bulgakov_utf8.txt` (роман Булгакова)
- **Векторная БД:** Chroma (in-memory)
- **Что демонстрирует:** то же, что `1_solar.py`, но на примере художественного произведения — деталей, которые модель может не знать

---

### 3. `3.1_load_from_github.py` — RAG с загрузкой данных из GitHub

- **Источник данных:** README.md файлы из репозиториев организации Elbrus-Bootcamp
- **Векторная БД:** PostgreSQL + PGVector
- **Что демонстрирует:**
  - Автоматическую загрузку данных через GitHub API (PyGithub)
  - Разбиение README по секциям Release
  - Rate limiting — контроль частоты запросов к API через декоратор `@retry`
  - Персистентное хранение в PostgreSQL (данные сохраняются между запусками)
  - RAG-цепочку через LCEL (`|` pipe syntax)

---

### 4. `3.2_create_exam.py` — RAG для генерации экзаменационных заданий

- **Источник данных:** Локальные Markdown-файлы из директории `./exams`
- **Векторная БД:** PostgreSQL + PGVector
- **Что демонстрирует:**
  - Загрузку и индексацию нескольких локальных файлов в цикле
  - Генерацию нового документа по образцу существующих
  - Сохранение результата в файл
- **Примечание:** Файлы в `./exams` создаются скриптом `3.1_load_from_github.py`

---

### 5. `4_extract_from_google_docs.py` — RAG с документом из Google Docs

- **Источник данных:** Google Drive документ (экспорт в text/plain)
- **Векторная БД:** Chroma (in-memory)
- **Что демонстрирует:**
  - OAuth2-авторизацию с Google и автоматическое обновление токена
  - Экспорт Google Docs через `files().export_media()`
  - Использование цепочки `RetrievalQA` для ответов на вопросы по документу

---

### 6. `5_web_cache.py` — RAG с PDF-документом

- **Источник данных:** `data/gotta-cache-em-all.pdf`
- **Векторная БД:** Chroma (in-memory)
- **Что демонстрирует:**
  - Загрузку PDF через `PyPDFLoader`
  - Разбиение с параметром `add_start_index=True` (метаданные с позицией в тексте)
  - Использование готового RAG-промпта из LangChain Hub

---

### 7. `6_get_files_from_drive.py` — Утилита для Google Drive

- **Что делает:** Показывает список файлов на Google Drive текущего пользователя
- **Требует:** `credentials.json` из Google Cloud Console

---

## Настройка окружения

### 1. Установка зависимостей

**Вариант A — pip (стандартный)**

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

**Вариант B — uv (быстрый, рекомендуется)**

[uv](https://docs.astral.sh/uv/) — современный менеджер пакетов и окружений для Python, работает в 10–100× быстрее pip.

```bash
# Установка uv (если ещё не установлен)
pip install uv

# Создание виртуального окружения и установка зависимостей одной командой
uv venv
uv pip install -r requirements.txt
```

Активация окружения, созданного через uv:

```bash
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### 2. Переменные окружения (файл `.env`)

```env
# GigaChat API
GIGACHAT_CREDENTIALS=your_gigachat_credentials

# PostgreSQL (для скриптов 3.1 и 3.2)
POSTGRES_USER=your_postgres_user
POSTGRES_PASSWORD=your_postgres_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=your_database_name

# GitHub API (для скрипта 3.1)
GITHUB_TOKEN=your_github_token

# Google Drive API (для скриптов 4 и 6)
# Файл credentials.json скачивается из Google Cloud Console
```

### 3. Дополнительная настройка

- **Google Drive API:** Создайте проект в [Google Cloud Console](https://console.cloud.google.com/), включите Drive API, скачайте `credentials.json`
- **PostgreSQL:** Установите PostgreSQL, создайте базу данных, включите расширение `pgvector`
- **GitHub API:** Создайте Personal Access Token с правами `repo:read` (или `public_repo` для публичных репозиториев)

---

## Запуск скриптов

### Базовые примеры RAG (без внешних зависимостей, кроме GigaChat)

```bash
# pip
python 1_solar.py      # RAG с данными о солнечной системе
python 2_bulgakov.py   # RAG с текстом Булгакова
python 5_web_cache.py  # RAG с PDF-документом

# uv
uv run 1_solar.py
uv run 2_bulgakov.py
uv run 5_web_cache.py
```

### Скрипты с PostgreSQL

```bash
# pip
python 3.1_load_from_github.py  # Шаг 1: загрузить данные с GitHub и наполнить БД
python 3.2_create_exam.py       # Шаг 2: сгенерировать экзамен из локальных файлов

# uv
uv run 3.1_load_from_github.py
uv run 3.2_create_exam.py
```

### Скрипты с Google Drive

```bash
# pip
python 6_get_files_from_drive.py       # Список файлов на Drive
python 4_extract_from_google_docs.py   # RAG по документу из Drive

# uv
uv run 6_get_files_from_drive.py
uv run 4_extract_from_google_docs.py
```

---

## Ключевые концепции RAG

### 1. Загрузка документов
Различные загрузчики LangChain для разных форматов:
- `TextLoader` — текстовые файлы (`.txt`, `.md`)
- `PyPDFLoader` — PDF-документы

### 2. Разбиение на чанки
`RecursiveCharacterTextSplitter` разбивает длинные тексты на небольшие перекрывающиеся фрагменты. Параметры:
- `chunk_size` — максимальный размер чанка в символах
- `chunk_overlap` — количество символов перекрытия между соседними чанками (контекст с границ не теряется)

### 3. Эмбеддинги
`GigaChatEmbeddings` преобразует каждый чанк в числовой вектор. Тексты с близким смыслом получают близкие векторы — это основа семантического поиска.

### 4. Векторные базы данных
| БД | Применение | Персистентность |
|----|------------|-----------------|
| Chroma (in-memory) | Прототипирование, небольшие объёмы | Нет (данные теряются при остановке) |
| PostgreSQL + PGVector | Продакшн, большие объёмы | Да |

### 5. Retrieval и Generation
1. Запрос пользователя преобразуется в вектор
2. В базе находятся k ближайших по смыслу чанков
3. Чанки + вопрос подставляются в промпт
4. Языковая модель генерирует ответ на основе контекста

---

## Пошаговое руководство по созданию RAG-системы

### Шаг 1: Подключение к GigaChat
```python
# Подключиться к GigaChat с использованием библиотеки langchain_gigachat.
# Загрузить переменные окружения из .env и создать экземпляр GigaChat.
```

### Шаг 2: Базовый запрос к модели
```python
# Отправить текстовый вопрос через HumanMessage и вывести ответ в консоль.
```

### Шаг 3: Загрузка и разбиение документов
```python
# Загрузить текстовый файл через TextLoader.
# Разбить на чанки с chunk_size=1000 и chunk_overlap=200.
```

### Шаг 4: Создание векторного индекса
```python
# Создать векторный индекс в Chroma на основе чанков и GigaChatEmbeddings.
```

### Шаг 5: Поиск похожих документов
```python
# Выполнить similarity_search по вопросу пользователя, получить k=4 результата.
```

### Шаг 6: Создание RAG-цепочки
```python
# Собрать цепочку: retriever + prompt + model.
# Передать найденные чанки как контекст и получить ответ модели.
```

### Шаг 7: Анализ результатов
```python
# Вывести количество чанков и содержимое первых 5 фрагментов для проверки разбиения.
```

---

## Дополнительные ресурсы

- [LangChain Documentation](https://python.langchain.com/)
- [Chroma Documentation](https://docs.trychroma.com/)
- [PGVector Documentation](https://github.com/pgvector/pgvector)
- [GigaChat API](https://developers.sber.ru/portal/products/gigachat)
- [Google Drive API](https://developers.google.com/drive/api/guides/about-sdk)
