кидаю тебе сразу цельный `llm_backbones_usage.md`, который можно положить рядом с предыдущим `.md` про бэкбоны.

---

````markdown
# LLM backbones — usage cheatsheet (transformers / vLLM / llama.cpp)

Файл для боя: минимальные, но рабочие паттерны:
- как загрузить модель,
- как дернуть `generate`,
- как встроить в RAG и JSON-валидатор.

Бэкбоны, на которые ориентируемся:
- Llama 3.1 8B Instruct
- Mistral 7B Instruct v0.3
- Gemma 2 9B IT
- Phi-3 Mini 4k Instruct (+ graph-версия как спец-инструмент)

---

## 0. Общие паттерны

### 0.1. Базовая обвязка для `transformers`

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_llm(
    model_id: str,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    use_flash_attention_2: bool = True,
):
    tok = AutoTokenizer.from_pretrained(model_id)
    # На всякий случай
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,   # часто нужно для HF-LLM
    )

    if use_flash_attention_2 and hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "flash_attention_2"

    model.eval()
    return tok, model


@torch.inference_mode()
def generate_text(
    tok,
    model,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 0.9,
    do_sample: bool = False,
    stop_tokens: list[str] | None = None,
):
    inputs = tok(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(model.device)

    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )

    text = tok.decode(generated[0], skip_special_tokens=True)
    if stop_tokens:
        for st in stop_tokens:
            if st in text:
                text = text.split(st)[0]
                break
    return text
````

**Дефолт на туре:**

* все, что про точный JSON / eval → `temperature=0.0`, `do_sample=False`;
* self-consistency → `temperature ~ 0.7`, `do_sample=True`, несколько прогонов.

---

## 1. transformers: конкретные бэкбоны

### 1.1. Llama 3.1 8B Instruct

```python
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
tok, model = load_llm(MODEL_ID, dtype=torch.bfloat16)

prompt = """You are a helpful assistant.
User: 2 + 2 = ?
Assistant:"""

answer = generate_text(tok, model, prompt, max_new_tokens=16)
print(answer)
```

Если используешь кванты:

```python
# Напр. FP4 от NVIDIA
MODEL_ID = "nvidia/Llama-3.1-8B-Instruct-FP4"
tok, model = load_llm(MODEL_ID, dtype=torch.float16)  # dtype тут не критичен, веса уже квантованы
```

---

### 1.2. Mistral 7B Instruct v0.3

```python
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
tok, model = load_llm(MODEL_ID, dtype=torch.bfloat16)

prompt = "<s>[INST] You are a helpful assistant. Answer briefly. What is 2+2? [/INST]"
print(generate_text(tok, model, prompt, max_new_tokens=16))
```

**Важно:** у Mistral часто свой формат промпта вида `[INST] ... [/INST]`. Можно зашить маленький helper:

```python
def mistral_inst(prompt: str, system: str | None = None) -> str:
    if system:
        return f"<s>[INST] {system}\n\n{prompt} [/INST]"
    return f"<s>[INST] {prompt} [/INST]"
```

---

### 1.3. Gemma 2 9B IT

```python
MODEL_ID = "google/gemma-2-9b-it"
tok, model = load_llm(MODEL_ID, dtype=torch.bfloat16)

def gemma_chat(system: str, user: str) -> str:
    # формат приблизительный, на туре можно слегка менять
    return f"<bos><start_of_turn>system\n{system}<end_of_turn>\n" \
           f"<start_of_turn>user\n{user}<end_of_turn>\n" \
           f"<start_of_turn>model\n"

prompt = gemma_chat(
    system="You are a helpful AI assistant.",
    user="Explain overfitting in one sentence.",
)
print(generate_text(tok, model, prompt, max_new_tokens=64))
```

Если берёшь INT4/FP8-квант, модельный ID меняется, но обвязка та же.

---

### 1.4. Phi-3 Mini 4k Instruct (+ graph)

```python
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
tok, model = load_llm(MODEL_ID, dtype=torch.bfloat16)

def phi_chat(system: str, user: str) -> str:
    return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"

prompt = phi_chat(
    "You are a concise assistant.",
    "What is the capital of France?",
)
print(generate_text(tok, model, prompt, max_new_tokens=16))
```

Для graph-версии (структурный вывод):

```python
MODEL_ID = "EmergentMethods/Phi-3-mini-4k-instruct-graph"
tok, model = load_llm(MODEL_ID, dtype=torch.bfloat16)

schema_hint = """
You MUST output a JSON with:
- "nodes": list of { "id": str, "label": str, "type": str }
- "edges": list of { "source": str, "target": str, "relation": str }
"""

text = "Alice works at OpenAI in San Francisco. Bob is Alice's manager."

prompt = phi_chat(
    f"You are an information extraction model. {schema_hint}",
    f"Extract a knowledge graph from this text: {text}",
)

json_str = generate_text(tok, model, prompt, max_new_tokens=256)
print(json_str)
```

---

## 2. RAG-паттерн на transformers

### 2.1. Построение prompt из кандидатов

```python
def build_rag_prompt(
    question: str,
    passages: list[str],
    system: str = "You are a helpful assistant that answers using the provided context only.",
) -> str:
    ctx = "\n\n".join(
        [f"[DOC {i+1}]\n{p}" for i, p in enumerate(passages)]
    )
    user = (
        "Answer the question using only the documents above. "
        "If the answer is not contained there, say you don't know.\n\n"
        f"Question: {question}"
    )
    # пример под Llama 3.1
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system}\n" \
           f"<|start_header_id|>user<|end_header_id|>\n{ctx}\n\n{user}\n" \
           f"<|start_header_id|>assistant<|end_header_id|>\n"


def rag_answer_llama(tok, model, question: str, passages: list[str]) -> str:
    prompt = build_rag_prompt(question, passages)
    return generate_text(
        tok,
        model,
        prompt,
        max_new_tokens=256,
        temperature=0.0,
        do_sample=False,
    )
```

---

## 3. JSON-валидатор / JSON-repair LLM

### 3.1. Общий промпт для “исправь JSON”

````python
import json

def json_repair_prompt(schema_description: str, bad_json: str) -> str:
    system = (
        "You are a strict JSON repair tool. "
        "You only output valid JSON that matches the given schema. "
        "Do not explain, do not add comments, do not wrap in markdown."
    )
    user = f"""
Schema (natural language description):
{schema_description}

Invalid or partially valid JSON candidate:
```json
{bad_json}
````

Return ONLY one valid JSON object that best matches both the schema and the candidate.
"""
# Пример формата под Mistral
return f"<s>[INST] {system}\n\n{user} [/INST]"

def repair_json_with_llm(tok, model, schema_description: str, bad_json: str) -> dict | None:
prompt = json_repair_prompt(schema_description, bad_json)
out = generate_text(tok, model, prompt, max_new_tokens=512)
# очень грубый парсинг: на туре можно добавить более аккуратный вырезатель фигурных скобок
try:
# ищем первый и последний символ '{' / '}'
start = out.find("{")
end = out.rfind("}")
if start == -1 or end == -1:
return None
json_str = out[start:end+1]
return json.loads(json_str)
except Exception:
return None

````

Типичный use-case:
- “большая” модель генерит JSON,
- ты валидируешь его `pydantic`/schema,
- если падает — кидаешь в `repair_json_with_llm` с маленькой Phi-3.

---

## 4. vLLM: быстрый inference

### 4.1. Общая обвязка

```python
from vllm import LLM, SamplingParams

def load_vllm(model_id: str, dtype: str = "bfloat16"):
    llm = LLM(
        model=model_id,
        dtype=dtype,                 # "bfloat16" / "float16"
        tensor_parallel_size=1,      # под одну GPU
        trust_remote_code=True,
    )
    return llm


def vllm_generate(
    llm,
    prompts: list[str],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 0.9,
    n: int = 1,
):
    params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        n=n,
    )
    outputs = llm.generate(prompts, params)
    # outputs[i].outputs[j].text — j-й сэмпл для i-го prompt
    res = []
    for out in outputs:
        res.append([o.text for o in out.outputs])
    return res
````

### 4.2. Пример с Llama 3.1

```python
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
llm = load_vllm(MODEL_ID, dtype="bfloat16")

prompt = "You are a helpful assistant.\nUser: 2 + 2 = ?\nAssistant:"
texts = vllm_generate(llm, [prompt], max_new_tokens=16, temperature=0.0)
print(texts[0][0])
```

### 4.3. RAG + vLLM, батчом

```python
questions = ["What is overfitting?", "Explain cross-validation shortly."]
passages_batch = [
    ["doc1_q1", "doc2_q1"],
    ["doc1_q2", "doc2_q2"],
]

prompts = [build_rag_prompt(q, ps) for q, ps in zip(questions, passages_batch)]
answers = vllm_generate(llm, prompts, max_new_tokens=256, temperature=0.0)
answers = [ans_list[0] for ans_list in answers]
```

---

## 5. llama.cpp (через `llama-cpp-python`)

Если на машине есть GGUF-квант (например, `phi-3-mini-4k-instruct-q4.gguf`), можно использовать `llama-cpp-python`:

```python
from llama_cpp import Llama

llm = Llama(
    model_path="phi-3-mini-4k-instruct-q4.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,   # все слои на GPU, если хватает памяти
    logits_all=False,
)

prompt = "<|system|>\nYou are a helpful assistant.\n<|user|>\n2+2?\n<|assistant|>\n"

out = llm(
    prompt,
    max_tokens=64,
    temperature=0.0,
    top_p=0.9,
    stop=["<|end|>"],
)
print(out["choices"][0]["text"])
```

Для RAG — просто подставляешь свой `build_rag_prompt` и передаёшь строку в `llm(...)`.

---

## 6. Self-consistency и reranking

### 6.1. Self-consistency

```python
def sc_generate(
    tok,
    model,
    prompt: str,
    num_samples: int = 5,
    max_new_tokens: int = 256,
):
    samples = []
    for _ in range(num_samples):
        txt = generate_text(
            tok,
            model,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        samples.append(txt)
    return samples

# потом делаешь:
# - majority vote (для классификации)
# - выбор ответа с max coverage/consistency по простым эвристикам
```

### 6.2. Использование маленькой модели как reranker/validator

Идея:

1. Большая модель генерит несколько кандидатов `answer_i`.
2. Маленькая (Phi-3 mini) получает prompt вида:

```text
Ты — реранкер. Дано задание, контекст и 3 кандидата ответа. 
Выбери один лучший и выведи его номер (1/2/3) и короткое объяснение.
```

3. Ты парсишь номер и берёшь соответствующий кандидат.

Это логикой уже живёт в коде пайплайна, сами бэкбоны те же.

---

## 7. Мини-cheatsheet выбора по runtime

* **Есть `transformers`, нет желания возиться с сервером →**
  бери `transformers` + `generate_text`.

* **Нужно много запросов и есть vLLM →**
  поднимаешь `vllm.LLM(...)`, всё катаешь батчами.

* **Нужен минимальный footprint и уже есть GGUF →**
  `llama-cpp-python` и GGUF-квант (Phi-3 / Gemma / Llama).

Главное: **не менять бэкбоны в последний момент на туре**. Лучше заранее отрепетировать по одному рабочему пайплайну под каждый стек (transformers / vLLM / llama.cpp) и дальше просто подставлять модельные ID под конкретную машину.

```

Если захочешь, могу отдельно собрать для тебя маленький `llm_utils.py`, где будут уже готовые функции типа `llm_answer_rag(...)`, `llm_repair_json(...)`, `llm_sc_classify(...)` под один-два выбранных бэкбона (например, Llama 3.1 + Phi-3).  

И напомню, на всякий случай: часть файлов, которые ты когда-то загружал в прошлых чатах (PDF/ноутбуки), уже недоступна — если нужно будет к ним вернуться, просто перезагрузи их сюда.
::contentReference[oaicite:0]{index=0}
```
