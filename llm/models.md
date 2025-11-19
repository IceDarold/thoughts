# LLM-бэкбоны, которые реально полезны на туре  
(с конкретными моделями на Hugging Face)

Цель: не каталог всего подряд, а короткий список **проверенных семейств**, из которых ты заранее знаешь, что и когда брать.

Ограничения, про которые держим в голове:

- ~20 GB VRAM (A100 20GB)
- Локальный inference (transformers / vLLM / llama.cpp / bitsandbytes)
- 7 часов тура → важна **скорость** и предсказуемость, а не только SOTA по LLM-лидерам

---

## 0. Роли LLM на туре

Типовые роли модели:

1. **Главный “мозг”**: отвечает на вопросы, пишет текст, решает reasoning-задачи, генерирует структурный JSON.  
2. **RAG-голова**: переписывает context → ответ, делает цитаты, проверяет факты.  
3. **Маленький “tool LLM”**: парсит схемы, чинит JSON, делает классификацию, self-consistency, reranking.  
4. (Опционально) **Спец-модель под extraction/graph**, если задача очень структурная.

Под каждую роль — свой удобный бэкбон.

---

## 1. Llama 3.1 8B Instruct — универсальный основной бэкбон

**Что это:**  
Новая волна Llama 3.1, 8B параметров, instruct-версия, хорошая как “одна главная модель на всё” (reasoning, код, RAG, JSON).  
- HF: `meta-llama/Llama-3.1-8B-Instruct` :contentReference[oaicite:0]{index=0}  

Есть готовые квантованные варианты:

- `nvidia/Llama-3.1-8B-Instruct-FP4` — FP4, очень экономно по VRAM. :contentReference[oaicite:1]{index=1}  
- `nvidia/Llama-3.1-8B-Instruct-FP8` — FP8, компромисс качество/ресурсы. :contentReference[oaicite:2]{index=2}  

**Когда брать на туре:**

- Хочешь **одну основную LLM “под всё”**: генерация текста, разбор условия, код, сложные промпты.
- Нужна нормальная **reasoning-способность** и адекватность в длинных промптах.
- Сценарии:
  - RAG: Llama как генератор поверх BM25/FAISS.
  - Задачи “прочитай PDF/описание → выдавай структурный JSON”.
  - Лёгкий кодинг / псевдокод / генерация сниппетов.

**Плюсы:**

- Сильный баланс: качество/адекватность/скорость.
- Огромное количество примеров в интернете, легко дебажить промпты.
- Квантованные FP4/FP8 версии влезают в 20GB VRAM вместе с приличным контекстом.

**Минусы:**

- Требует согласия на лицензию Meta на HF; на туре это решают организаторы (в идеале — веса уже лежат локально). :contentReference[oaicite:3]{index=3}  

---

## 2. Mistral 7B Instruct v0.3 — компактный, быстрый, с function calling

**Что это:**  
Лёгкая 7B-модель, сильно оптимизированная, с хорошей speed/quality. Есть instruct-версия с function calling.  
- HF (официальный/комьюнити):  
  - `mistralai/Mistral-7B-v0.3` — base. :contentReference[oaicite:4]{index=4}  
  - `mistralai/Mistral-7B-Instruct-v0.3` / `mistral-community/Mistral-7B-Instruct-v0.3`. :contentReference[oaicite:5]{index=5}  

Есть и готовые 4-bit-варианты для дешёвого запуска:

- `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` — сразу quantized под bitsandbytes. :contentReference[oaicite:6]{index=6}  

**Когда использовать:**

- Нужна **быстрая, но умная** модель для:
  - JSON/структурного вывода,
  - function calling (интеграция с tool-логикой),
  - легкого RAG без супер-длинных контекстов.
- Хочешь **экономить VRAM и время**, но не хочешь уходить в совсем малышей.

**Плюсы:**

- Очень хорошая скорость на 7B.
- Из коробки есть function calling → удобно для задач “выведи строго такой JSON/список шагов”.
- Отличный кандидат на роль “рабочей лошадки” для средних задач.

**Минусы:**

- Чуть слабее Llama 3.1 / Gemma 2 в сложном reasoning, но зато легче и проще по ресурсам.

---

## 3. Gemma 2 9B IT — сильный “качество-текст” бэкбон

**Что это:**  
Gemma 2 — семейство от Google, 2B/9B; 9B-IT — instruction-tuned вариант. Хорошая генерация текста, сильный reasoning, хорошая мультилингва.  
- HF: `google/gemma-2-9b-it` и `google/gemma-2-9b-it-pytorch`. :contentReference[oaicite:7]{index=7}  

Популярные кванты:

- `hugging-quants/gemma-2-9b-it-AWQ-INT4` — INT4 через AutoAWQ, удобно под GPU с ограниченной памятью. :contentReference[oaicite:8]{index=8}  
- `RedHatAI/gemma-2-9b-it-FP8` — FP8 версия, оптимизирована под vLLM >= 0.5.1. :contentReference[oaicite:9]{index=9}  
- `bartowski/gemma-2-9b-it-GGUF` — пачка GGUF-квантов под llama.cpp. :contentReference[oaicite:10]{index=10}  
- Есть и unsloth-вариант: `unsloth/gemma-2-9b-it` для быстрой донастройки. :contentReference[oaicite:11]{index=11}  

**Когда использовать:**

- Хочешь **максимум качества** при ещё жизнеспособном размере (9B).
- Нужен **сильный ответчик для RAG**: аккуратный тон, хорошая логическая связность.
- Задачи:  
  - сложные генеративные — суммаризация + структурирование,  
  - длинные цепочки рассуждений,  
  - аккуратный JSON/Tool Output.

**Плюсы:**

- Отличное качество текста и reasoning.
- С квантом (INT4/FP8) реально запускается в 20GB VRAM с нормальным контекстом.
- Хорошие варианты под разные фреймворки: transformers, vLLM, llama.cpp.

**Минусы:**

- Чуть тяжелее по параметрам и памяти, чем Llama 3.1 8B / Mistral 7B.
- Лицензия Google Gemma — смотри заранее, что именно организаторы положили в образ.

---

## 4. Phi-3 Mini 4K Instruct — маленький, но умный “tool LLM”

**Что это:**  
Семейство компактных моделей от Microsoft. Phi-3-mini — ~3.8B, обучен на качественном, reasoning-ориентированном датасете, работает удивительно хорошо для своего размера.  
- HF: `microsoft/Phi-3-mini-4k-instruct` :contentReference[oaicite:12]{index=12}  

Есть разные варианты:

- ONNX-версии: `microsoft/Phi-3-mini-4k-instruct-onnx` и друзья — удобно под CPU / ускоренный inference. :contentReference[oaicite:13]{index=13}  
- GGUF: `microsoft/Phi-3-mini-4k-instruct-gguf` — под llama.cpp, есть уже q4 и др. квантования. :contentReference[oaicite:14]{index=14}  
- Unsloth: `unsloth/Phi-3-mini-4k-instruct` — quantized 4bit, удобен для дообучения. :contentReference[oaicite:15]{index=15}  

Есть даже специализированный fine-tune:

- `EmergentMethods/Phi-3-mini-4k-instruct-graph` — под извлечение entity-relationship графов (структурный JSON/graph из текста). :contentReference[oaicite:16]{index=16}  

**Когда использовать:**

- Как **вторую, маленькую модель “под капотом”**:
  - быстрый JSON-repair / валидация,
  - классификация/тэггинг,
  - self-consistency (прогнать 5 вариантов/majority vote),
  - дешёвый reranker.
- Когда надо **масштабировать число запросов** (например, много мелких промптов в пайплайне).

**Плюсы:**

- Очень дёшево по VRAM/CPU, летает даже в 4bit.
- Отлично подходит как “LLM-инструмент”, а не главный генератор.
- Есть спец-модель под граф-экстракцию, что в задачах типа “собери knowledge graph / event graph” может дать бонус.

**Минусы:**

- Не главный кандидат для сложных open-ended ответов (статьи, длинные reasoning-цепочки).
- Нужно тщательнее дизайнить промпты, чтобы не получить халтуру.

---

## 5. Спец-файнтюны под структурный вывод / graph

Если задача на туре окажется сильно структурной (event-graph, KG, логические связи), можно рассмотреть:

- `EmergentMethods/Phi-3-mini-4k-instruct-graph` — fine-tune Phi-3-mini для извлечения entity-relationship графов, заточен под генерацию связей на уровне GPT-4 качества при меньших ресурсах. :contentReference[oaicite:17]{index=17}  

Типичный сценарий на туре:

1. Основной RAG/ответчик: Llama 3.1 / Gemma 2.  
2. Вспомогательный “graph-LLM”: Phi-3-graph  
   - из текста / контекста RAG вытаскивает “nodes + edges” в JSON,  
   - потом табличка/graph-постпроцессинг делает метрику.

---

## 6. Как это всё разложить под роли

### 6.1 Роль “главный ответчик / RAG-голова”

**Рекомендуемые бэкбоны:**

- `meta-llama/Llama-3.1-8B-Instruct` (в FP4/FP8 вариантах от NVIDIA). :contentReference[oaicite:18]{index=18}  
- `google/gemma-2-9b-it` (в INT4/FP8/ GGUF вариантах). :contentReference[oaicite:19]{index=19}  

**Тип задач:**

- RAG с длинным контекстом (PDF/HTML → answer + citations).
- Сложные текстовые задачи: план, рассуждение, объяснение решения.
- Сложный структурный JSON, где важна именно логика, а не только формат.

---

### 6.2 Роль “быстрый универсал” (JSON, tools, function calling)

**Рекомендуемый бэкбон:**

- `mistralai/Mistral-7B-Instruct-v0.3` (+ 4bit-версии). :contentReference[oaicite:20]{index=20}  

**Тип задач:**

- Аккуратный JSON-вывод, function calling (если ты сам реализуешь tool-луп).
- Быстрые короткие промпты в пайплайне (вызовов много, latency критичен).
- Вспомогательные шаги: ранкинг вариантов, генерация intermediate reasoning.

---

### 6.3 Роль “маленький tool LLM / self-consistency engine”

**Рекомендуемый бэкбон:**

- `microsoft/Phi-3-mini-4k-instruct` (или его 4bit/GGUF-версии). :contentReference[oaicite:21]{index=21}  

**Тип задач:**

- Валидация JSON:  
  промпт вида «вот schema, вот candidate → либо верни исправленный JSON, либо напиши error».
- Классификация/тэггинг/intent detection.
- Прогон много маленьких self-consistency итераций (N=5–10), а потом агрегировать.

---

## 7. Мини-шпаргалка “что брать по умолчанию”

Если надо быстро принять решение на туре:

- **Одна большая модель (главная):**
  - Llama: `meta-llama/Llama-3.1-8B-Instruct` (в FP4/FP8-кванте). :contentReference[oaicite:22]{index=22}  
  - или Gemma: `google/gemma-2-9b-it` (в INT4/FP8/ GGUF). :contentReference[oaicite:23]{index=23}  

- **Второй, лёгкий помощник:**
  - `microsoft/Phi-3-mini-4k-instruct` как tool LLM и JSON-ремонтник. :contentReference[oaicite:24]{index=24}  

- **Альтернативный универсал (если Llama/Gemma недоступны):**
  - `mistralai/Mistral-7B-Instruct-v0.3`. :contentReference[oaicite:25]{index=25}  

---

## 8. На что обратить внимание заранее

1. **Лицензии и доступ:**  
   Llama 3.1 и Gemma 2 требуют соглашения на HF, скачивание через токен — на туре важно, чтобы веса были **предустановлены** (или чтобы это явно было указано в регламенте).

2. **Кванты под твой стек:**
   - Для `transformers + bitsandbytes`: bnb-4bit/8bit (unsloth / bnb-версии). :contentReference[oaicite:26]{index=26}  
   - Для `llama.cpp`: GGUF (`…-GGUF`, `…-q4.gguf`). :contentReference[oaicite:27]{index=27}  
   - Для `vLLM`: FP8/F16-версии, подготовленные под vLLM (например, FP8 Gemma 2 от RedHatAI). :contentReference[oaicite:28]{index=28}  

3. **Контекст:**  
   Проверь заранее, какое **максимальное окно контекста** у каждой модели (Phi-3-mini 4k vs 128k, Llama 3.1, Gemma 2) и соотнеси с потенциальными задачами (длинный RAG vs короткие snippet’ы). :contentReference[oaicite:29]{index=29}  

---

Идея простая: ты не хочешь на туре разбираться в зоопарке из 15 LLM.  
Ты хочешь приехать уже с внутренней картой:

- **“Вот мой основной мозг”** (Llama/Gemma).  
- **“Вот быстрый универсал”** (Mistral).  
- **“Вот маленький инструмент”** (Phi-3-mini, плюс, если надо, graph-версия).

Дальше уже играешь пайплайнами, а не угадыванием моделей.

