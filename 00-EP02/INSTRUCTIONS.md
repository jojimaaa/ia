# Instruções para Implementação - Projeto de VQA Multimodal (Kaggle)

## Objetivo
Desenvolver um **único Jupyter Notebook** para Google Colab (ou ambiente local leve) que implemente um pipeline completo para um desafio de **Visual Question Answering (VQA)**. O foco é demonstrar a aplicação de técnicas modernas de otimização de LLMs/VLMs em inferência e adaptação.

---

## 1. Configuração e Parâmetros Globais
Responsável por instalar bibliotecas (`transformers`, `accelerate`, `bitsandbytes`, `peft`), montar o Google Drive (se no Colab) e definir parâmetros centrais.

**Parâmetros recomendados:**

```python
DEBUG = True

# Modelo para o Colab Free:
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
USE_4BIT = True

# Modelo de fallback para PC local (CPU/Pouca RAM):
# MODEL_NAME = "vikhyatk/moondream2"

CHECKPOINT_INTERVAL = 50
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7

# Validação local para não depender apenas do Kaggle
VALIDATION_SPLIT_RATIO = 0.1 # 10% de 2000 = 200 amostras para validação
```

---

## 2. Preparação do Dataset e Validação Local
- Localizar a pasta `/kaggle_dataset/` (treino e teste).
- Carregar os dados usando `pandas` ou HuggingFace `datasets`.
- Dividir o `train_dataset` em conjuntos de `train` (1.800 amostras) e `val` (200 amostras).
- Nos experimentos 1 a 5, **avaliar o modelo primeiro no conjunto `val`**. Apenas se o resultado for satisfatório, rodar no `test_dataset` para gerar o arquivo de submissão do Kaggle.

---

## 3. Funções Auxiliares e Gerenciamento de Memória
Criar funções reutilizáveis, com forte foco em resiliência e parser de texto:

- `generate_submission(...)` - Pode ser aproveitada do módulo em `data/generate_submission`
- `prepare_dataset(...)` - Pode ser aproveitada do módulo em `data/prepare_dataset`
- **`limpar_memoria()`**: Função contendo `import gc; gc.collect(); torch.cuda.empty_cache()`. Deve ser chamada frequentemente para evitar Out of Memory (OOM).
- **`extrair_resposta_final(texto_gerado)`**: Função robusta com RegEx para encontrar blocos como `Final Answer:`, ignorando o raciocínio gerado pelo CoT ou Reflexion.
- `salvar_checkpoint()` e `carregar_checkpoint()`: Para garantir a retomada em caso de queda do ambiente.

---

## 4. Estrutura dos Experimentos (Independência Total)
Cada experimento deve ser uma seção isolada que começa executando `limpar_memoria()`, verificando checkpoints existentes e calculando a acurácia no conjunto `val`.

### Experimento 1: Baseline (Zero-Shot)
- **Prompt:** O mais simples possível. "Responda à pergunta baseada na imagem de forma curta e direta."
- **Objetivo:** Estabelecer o piso de acurácia.
- **Saída:** `submission_baseline.csv`

### Experimento 2: Prompt Engineering
- **Prompt:** Instruções sistemáticas de sistema (System Prompt). Ex: "Você é um especialista em análise visual. Siga as regras: 1. Seja conciso. 2. Foque no objeto central. 3. Não dê explicações, apenas a resposta final."
- **Saída:** `submission_prompt.csv`

### Experimento 3: Chain of Thought (CoT)
- **Prompt:** Modificado para forçar raciocínio. Ex: "Descreva os passos para chegar à resposta. Pense passo a passo (Think step-by-step). Finalize sua resposta obrigatoriamente com 'Final Answer: [resposta]'."
- **Ponto de Atenção:** Passar a saída pela função `extrair_resposta_final()`. Apenas o conteúdo extraído vai para o CSV.
- **Saída:** `submission_cot.csv`

### Experimento 4: Self Consistency
- **Técnica:** Para cada amostra, rodar a inferência **3 vezes** com `TEMPERATURE = 0.7`.
- **Agregação:** Implementar uma função de *Majority Voting* (Voto da Maioria). A resposta que mais aparecer entre as 3 tentativas é a escolhida.
- **Atenção:** Testar exaustivamente no modo `DEBUG = True` antes de rodar completo, pois exige 3x mais processamento.
- **Saída:** `submission_self_consistency.csv`

### Experimento 5: Test-Time Compute (TTC) - Reflexion
- **Fluxo de 2 Etapas (Pipeline):**
  1. **Geração:** Pedir a resposta base.
  2. **Reflexão:** Enviar a mesma imagem, a pergunta original e a resposta gerada de volta ao modelo com o prompt: *"Você respondeu [Resposta] para a pergunta [Pergunta]. Olhe para a imagem novamente. Há algum erro na sua análise? Corrija e forneça a resposta definitiva no formato 'Final Answer: [correção]'."*
- **Saída:** `submission_ttc.csv`

### Experimento 6: Parameter-Efficient Fine-Tuning (QLoRA) - *Opcional*
- **Técnica:** QLoRA (Quantized LoRA) utilizando o conjunto `train` (1.800 imagens).
- **Configurações estritas** (para Colab Free):
  - `batch_size = 1`
  - `gradient_accumulation_steps = 4` ou `8`
  - `r = 8` (Rank do LoRA), `alpha = 16`.
  - Treinar apenas os pesos de atenção (`q_proj`, `v_proj`).
- Salvar os adaptadores `adapter_model.safetensors`, recarregar no modelo base e inferir no `test_dataset`.
- **Saída:** `submission_lora.csv`

---

## 5. Comparação e Resultados
Criar uma seção final comparando todos os experimentos através de um DataFrame (Pandas):

| Experimento         | Acurácia Validação | Tempo Médio/Imagem | Custo Computacional |
|---------------------|--------------------|--------------------|---------------------|
| Baseline            |                    |                    | Baixo               |
| Prompt Eng.         |                    |                    | Baixo               |
| CoT                 |                    |                    | Médio               |
| Self Consistency    |                    |                    | Alto (3x)           |
| TTC (Reflexion)     |                    |                    | Alto (2x)           |
| QLoRA               |                    |                    | Muito Alto (Treino) |

Discutir qualitativamente as vantagens, limitações e o impacto real de cada técnica na qualidade das respostas geradas pelo modelo.
