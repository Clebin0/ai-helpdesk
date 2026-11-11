# Fundamentos de LLM com Ollama

Laboratório introdutório para entender chamadas a modelos de linguagem sem depender de créditos de API. O código usa HTTP da biblioteca padrão do Python e um modelo executado localmente pelo Ollama.

## O que você vai praticar

1. **Chat completion:** diferença entre mensagens `system` e `user`, temperatura e resposta do modelo.
2. **Structured output:** solicitar JSON usando JSON Schema e validar o retorno com Pydantic.
3. **Confiabilidade:** limites de entrada e resposta, tratamento de erros, validação de tipos e testes sem chamar o modelo real.
4. **Segurança de prompt:** o texto de um chamado é tratado como dado não confiável; o exemplo não executa ações nem substitui a análise humana.

## Requisitos

- Python 3.12 ou superior.
- Ollama instalado e em execução.
- Modelo local disponível. O padrão deste laboratório é `qwen3.5:4b-q4_K_M`.

Confira o serviço e os modelos:

```powershell
ollama list
```

Caso o modelo ainda não esteja instalado:

```powershell
ollama pull qwen3.5:4b-q4_K_M
```

O download inicial exige internet e espaço em disco. Depois, a inferência pode ser executada localmente, sem cobrança por token de API.

## Executar a partir da raiz do repositório

Ative o ambiente virtual usado pelo projeto e instale as dependências:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 1. Chat básico

```powershell
python -m examples.llm_foundations.chat "Explique a diferença entre token e contexto em um LLM."
```

A chamada padrão limita a geração a 160 tokens e envia `think=false` para evitar raciocínio prolongado em perguntas operacionais curtas. O limite pode ser ajustado no código pela opção `num_predict` (1 a 4.096).

Você também pode executar sem argumento para digitar a pergunta no terminal:

```powershell
python -m examples.llm_foundations.chat
```

### 2. Saída estruturada

```powershell
python -m examples.llm_foundations.ticket_analysis "Usuários do escritório não conseguem resolver nomes DNS desde as 10h."
```

O programa pede ao Ollama um objeto JSON compatível com o schema de `TicketAnalysis` e valida o resultado. JSON válido não significa análise correta: confira categoria, prioridade e ação sugerida antes de usar a informação.

### 3. Embeddings e busca semântica

```powershell
python -m examples.llm_foundations.rag_search "Alguns computadores não conseguem encontrar os servidores pelo nome."
```

O comando gera embeddings locais para os documentos sintéticos e para a pergunta, compara os vetores por similaridade de cosseno e mostra os trechos recuperados com fonte e índice. A primeira execução pode baixar o modelo de embeddings. Esta etapa é apenas recuperação: ela ainda não chama o LLM para redigir uma resposta.

```text
Documentos → chunks → embeddings → índice
                                   ↑
Pergunta → embedding → similaridade → fontes recuperadas
```

Parâmetros opcionais:

```powershell
python -m examples.llm_foundations.rag_search "falha de DNS" --top-k 2 --min-score 0.25
```

O limite de similaridade não é uma probabilidade de acerto. Ajuste-o com perguntas de avaliação e confira as fontes recuperadas.

### Configuração opcional

O padrão é `http://127.0.0.1:11434` e `qwen3.5:4b-q4_K_M`. Para alterar sem editar o código:

```powershell
$env:OLLAMA_HOST = "http://127.0.0.1:11434"
$env:OLLAMA_MODEL = "qwen3.5:4b-q4_K_M"
```

## Testes

Os testes usam respostas simuladas e não exigem que o Ollama esteja ativo:

```powershell
python -m pytest tests/test_llm_foundations.py -q
```

## Estrutura

```text
examples/
  llm_foundations/
    ollama_client.py      # cliente HTTP local e validação de entrada
    chat.py               # primeira chamada de chat
    ticket_analysis.py    # JSON Schema + validação Pydantic
tests/
  test_llm_foundations.py # testes sem rede e sem inferência
```

## Limites e segurança

- O cliente não executa comandos, ferramentas ou ações externas.
- Não envie dados pessoais, credenciais, tokens, logs privados ou documentos sem autorização.
- A saída do LLM é conteúdo não confiável; valide o JSON e revise as sugestões.
- O endpoint de Ollama deve permanecer acessível somente a clientes autorizados na máquina ou rede confiável.
- O projeto demonstra fundamentos; não é uma política de atendimento pronta para produção.

## Referência técnica

O parâmetro `format` da API `/api/chat` do Ollama aceita o modo `json` ou um JSON Schema para saídas estruturadas: [documentação oficial da API](https://github.com/ollama/ollama/blob/main/docs/api.md).
