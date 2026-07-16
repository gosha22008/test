"""Агент по внутренней документации: LLM сама решает, когда искать.

Реализует цикл tool-calling — то, что в готовом виде делает агентная платформа:
модель получает описание инструмента, решает вызвать его, наш код выполняет
вызов и возвращает результат модели, модель формулирует ответ.

Запуск (Ollama и agent-api должны быть подняты):
    cd src && python ask.py "какой срок согласования отпуска?"
"""

import os
import sys

import httpx
import yaml

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:14b")
SEARCH_URL = os.getenv("SEARCH_URL", "http://localhost:8000/search")
AGENT_CONFIG = os.getenv("AGENT_CONFIG", "../agent.yaml")
MAX_ITERATIONS = 5


# Описание инструмента для модели. По полю description она решает,
# КОГДА его звать; по parameters — какие аргументы сгенерировать.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Поиск по внутренним регламентам компании. Вызывай для каждого "
                "содержательного вопроса, чтобы получить фрагменты документов "
                "с указанием источника."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


def load_system_prompt() -> str:
    """
    Читает системный промпт из agent.yaml.

    Промпт хранится в одном месте: тот же файл описывает агента для платформы.
    Дублировать текст в коде нельзя — версии разъедутся.
    """
    with open(AGENT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)["system_prompt"]


def search_knowledge_base(query: str) -> str:
    """
    Инструмент агента: вызывает /search и форматирует чанки для модели.

    Каждый фрагмент подписывается источником — без этого модель не сможет
    сослаться на документ и страницу. Пустой результат тоже валиден: по нему
    агент обязан честно сказать, что информации нет.
    """
    response = httpx.post(SEARCH_URL, json={"query": query}, timeout=60.0)
    response.raise_for_status()
    results = response.json()["results"]

    if not results:
        return "Ничего не найдено."

    chunks = [
        f"[Фрагмент {i}] Источник: {r['file_name']}, стр. {r['page']}\n{r['text']}"
        for i, r in enumerate(results, start=1)
    ]
    return "\n---\n".join(chunks)


def chat(messages: list[dict]) -> dict:
    """
    Один вызов LLM со списком доступных инструментов.

    think=False отключает рассуждения Qwen3 вслух: агенту по документации
    нужна скорость и фактичность, а не размышления на страницу.
    """
    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": messages,
            "tools": TOOLS,
            "think": False,
            "stream": False,
            "options": {"temperature": 0.1, "seed": 42},
        },
        timeout=180.0,
    )
    response.raise_for_status()
    return response.json()["message"]


def ask(question: str, verbose: bool = True) -> str:
    """
    Цикл агента: подумать -> вызвать инструмент -> прочитать результат -> ответить.

    Модель сама решает, нужен ли поиск и как переформулировать запрос.
    Цикл крутится, пока модель не вернёт текст без вызовов инструментов;
    MAX_ITERATIONS страхует от зацикливания.
    """
    messages = [
        {"role": "system", "content": load_system_prompt()},
        {"role": "user", "content": question},
    ]

    for step in range(1, MAX_ITERATIONS + 1):
        message = chat(messages)
        messages.append(message)

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            return message["content"]  # текст без вызовов = финальный ответ

        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]

            if verbose:
                print(f"[шаг {step}] {name}({args})")

            if name == "search_knowledge_base":
                result = search_knowledge_base(args["query"])
            else:
                result = f"Инструмент {name} не существует."

            # Результат возвращается модели отдельным сообщением с ролью tool
            messages.append({"role": "tool", "content": result})

    return "Превышен лимит итераций."


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Использование: python ask.py "ваш вопрос"')
        sys.exit(1)

    print(f"\n{ask(' '.join(sys.argv[1:]))}")
