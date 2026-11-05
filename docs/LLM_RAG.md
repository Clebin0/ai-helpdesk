# AI Helpdesk — copiloto RAG local

O classificador original TF-IDF continua separado e inalterado. O módulo /assistant adiciona busca semântica, recuperação de documentos e geração de respostas citadas.

## Arquitetura padrão (sem cobrança por chamada)

1. Um mantenedor envia documentação aprovada para POST /assistant/ingest.
2. O FastEmbed gera embeddings localmente, na CPU, usando um modelo multilíngue.
3. Textos, fontes e vetores são armazenados no SQLite local.
4. O usuário envia uma pergunta para POST /assistant/chat.
5. A pergunta é vetorizada localmente; o sistema calcula similaridade por cosseno e descarta resultados pouco relevantes.
6. O Ollama executa Qwen 3.5 4B localmente e responde com base nos trechos recuperados, citando referências como [S1].

O texto dos documentos e das perguntas não precisa ser enviado a um provedor externo no modo padrão. Depois de baixar os modelos, embeddings e geração de resposta são processados localmente. Não há cobrança por token; o custo é o uso de energia e recursos do próprio computador.

## Modelos usados

- Chat: qwen3.5:4b-q4_K_M via Ollama local.
- Embeddings: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 via FastEmbed/ONNX, local e multilíngue.
- Contexto do Ollama: 4096 tokens por padrão; ajuste OLLAMA_NUM_CTX se necessário.

## Requisitos e instalação

O modelo de chat ocupa aproximadamente 3,3 GB em disco. O modelo de embeddings ocupa cerca de 220 MB. O desempenho de geração depende do processador e da memória; em máquinas sem GPU compatível, o modelo roda na CPU e pode responder mais lentamente.

Instale o Ollama para Windows a partir de https://ollama.com/download/windows. Depois, inicie o serviço Ollama e baixe o modelo:

    ollama pull qwen3.5:4b-q4_K_M

Verifique se o serviço responde em http://127.0.0.1:11434.

## Configuração local (PowerShell)

Copie o exemplo e edite o arquivo local:

    Copy-Item .env.example .env
    notepad .env

No modo padrão, mantenha:

    LLM_PROVIDER=ollama
    OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
    OLLAMA_CHAT_MODEL=qwen3.5:4b-q4_K_M
    OLLAMA_NUM_CTX=4096
    RAG_EMBEDDING_PROVIDER=local

Preencha também ASSISTANT_ACCESS_TOKEN e ASSISTANT_ADMIN_TOKEN com valores independentes. Não reutilize credenciais de provedores. Para gerar cada token:

    python -c "import secrets; print(secrets.token_urlsafe(32))"

Depois de editar o .env, carregue as variáveis na sessão do PowerShell que iniciará a API:

    Get-Content .env | ForEach-Object {
      if ($_ -match '^\s*([^#=]+)\s*=(.*)$') {
        [Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim(), 'Process')
      }
    }

Esse comando não imprime os valores.

## Executar a API

    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    python train.py
    uvicorn app:app --reload --port 8001

Abra http://127.0.0.1:8001/docs.

A aplicação espera que o serviço Ollama esteja em execução. A API local do Ollama é compatível com o cliente OpenAI em http://127.0.0.1:11434/v1; o valor da chave é ignorado pelo servidor local.

## Provedores opcionais

O Hugging Face continua disponível como alternativa:

    LLM_PROVIDER=huggingface
    HF_TOKEN=seu_token
    HF_CHAT_MODEL=Qwen/Qwen3.5-9B:cheapest

O token precisa da permissão Make calls to Inference Providers. O roteador OpenAI-compatible da Hugging Face atende chat completions; os embeddings continuam locais. A conta testada retornou HTTP 402 informando que é necessário comprar créditos pré-pagos ou ter créditos incluídos em plano PRO. Por esse motivo, o modo Hugging Face não é o padrão e não deve ser tratado como gratuito para essa conta.

Para usar OpenAI em uma instalação com créditos, defina LLM_PROVIDER=openai, RAG_EMBEDDING_PROVIDER=openai e configure OPENAI_API_KEY e os nomes de modelos.

Ao mudar o modelo de embedding, os vetores de outro modelo não são comparáveis. Reindexe os documentos que deseja consultar com o novo modelo; os vetores antigos são mantidos sob o namespace anterior.

## Indexar um documento

Envie apenas documentação que você tem autorização para usar. Não envie dados pessoais, registros de clientes, senhas, tokens, códigos MFA, segredos ou incidentes sem anonimização.

    $body = @{
      source = 'network-dns-runbook.md'
      content = 'DNS: confirme usuários afetados, endereço IP, gateway e resolvedores configurados. Teste a resolução com nslookup e registre a mensagem de erro. Não altere configurações de produção sem aprovação.'
    } | ConvertTo-Json

    Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8001/assistant/ingest' -Headers @{ 'X-Admin-Token' = $env:ASSISTANT_ADMIN_TOKEN } -ContentType 'application/json' -Body $body

## Fazer uma pergunta

    $body = @{ question = 'Como começo a diagnosticar uma falha de resolução DNS?'; top_k = 4 } | ConvertTo-Json

    Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8001/assistant/chat' -Headers @{ 'X-Assistant-Token' = $env:ASSISTANT_ACCESS_TOKEN } -ContentType 'application/json' -Body $body

A resposta contém texto, nomes dos modelos e fontes citadas. O modelo é orientado a declarar quando não há informação suficiente. Confira os trechos originais antes de executar procedimentos.

## Endpoints e segurança

| Método | Rota | Acesso |
|---|---|---|
| GET | /assistant/health | Público; apenas estado de configuração |
| GET | /assistant/documents | X-Admin-Token |
| POST | /assistant/ingest | X-Admin-Token |
| POST | /assistant/chat | X-Assistant-Token |

O conteúdo recuperado é dado não confiável. Instruções inseridas em documentos não podem mudar o papel do assistente. Ele não solicita senhas, tokens ou códigos MFA e não executa ações em sistemas.

Mantenha a API em rede confiável. Antes de expor para terceiros, use HTTPS, controle de identidade e proteção contra abuso.

## Limites conhecidos

- Busca linear em SQLite; avalie pgvector ou banco vetorial dedicado para escalar.
- Chunking baseado em caracteres, não tokens.
- O limiar RAG_MIN_RELEVANCE exige avaliação com um conjunto de perguntas reais.
- Sem memória de conversa, busca híbrida, reranker ou execução de ferramentas.
- Faltam avaliações sistemáticas de precisão/recall, qualidade de citações, abstenção e latência. Não classifique o módulo como pronto para produção sem essas avaliações.
