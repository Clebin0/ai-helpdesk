# AI Helpdesk — copiloto RAG local

O classificador TF-IDF continua separado e inalterado. O módulo /assistant adiciona busca semântica, recuperação de documentos e geração de respostas com fontes.

## Arquitetura padrão

1. Um mantenedor envia documentação aprovada para POST /assistant/ingest.
2. O FastEmbed gera embeddings localmente, na CPU, usando um modelo multilíngue.
3. Textos, fontes e vetores são armazenados no SQLite local.
4. O usuário pergunta pelo chat em /assistant/ui ou via API.
5. A pergunta (e os últimos turnos do chat, quando houver) é vetorizada localmente; o sistema calcula similaridade por cosseno e descarta resultados pouco relevantes.
6. O Ollama executa Qwen 3.5 localmente. As respostas são transmitidas em streaming e citam referências como [S1].

No modo padrão, o texto dos documentos, perguntas e respostas não precisa ser enviado a um provedor externo. Depois de baixar os modelos, embeddings e geração de resposta são processados localmente. Não há cobrança por token; o custo é o uso de energia e recursos do próprio computador.

## Modelos e otimizações

- Chat: qwen3.5:4b-q4_K_M via Ollama local.
- Embeddings: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 via FastEmbed/ONNX.
- Contexto do Ollama: 2048 tokens por padrão.
- Saída máxima: 160 tokens por resposta por padrão.
- Think: false para evitar raciocínio oculto desnecessário em tarefas operacionais curtas.
- Keep-alive: 30m para evitar recarregar o modelo entre mensagens.
- Streaming: o endpoint /assistant/chat/stream entrega fragmentos conforme são gerados.

Os valores podem ser ajustados por OLLAMA_NUM_CTX, OLLAMA_NUM_PREDICT, OLLAMA_KEEP_ALIVE e OLLAMA_NUM_THREADS no .env. Aumentar o contexto ou a quantidade de tokens pode aumentar o tempo de resposta e o consumo de memória.

## Requisitos e instalação

O modelo de chat ocupa aproximadamente 3,3 GB em disco. O modelo de embeddings ocupa cerca de 220 MB. Em máquinas sem GPU compatível, o Qwen roda na CPU e pode responder mais lentamente.

Instale o Ollama para Windows em https://ollama.com/download/windows. Depois, inicie o serviço e baixe o modelo:

    ollama pull qwen3.5:4b-q4_K_M

Confirme que o serviço responde em http://127.0.0.1:11434.

## Configuração local (PowerShell)

Copie o exemplo e edite o arquivo local:

    Copy-Item .env.example .env
    notepad .env

No modo padrão, mantenha:

    LLM_PROVIDER=ollama
    OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
    OLLAMA_CHAT_MODEL=qwen3.5:4b-q4_K_M
    OLLAMA_NUM_CTX=2048
    OLLAMA_NUM_PREDICT=160
    OLLAMA_KEEP_ALIVE=30m
    OLLAMA_NUM_THREADS=6
    RAG_EMBEDDING_PROVIDER=local

Preencha ASSISTANT_ACCESS_TOKEN e ASSISTANT_ADMIN_TOKEN com valores independentes. Não reutilize credenciais de provedores. Para gerar cada token:

    python -c "import secrets; print(secrets.token_urlsafe(32))"

Depois de editar o .env, carregue as variáveis na sessão do PowerShell que iniciará a API:

    Get-Content .env | ForEach-Object {
      if ($_ -match '^\s*([^#=]+)\s*=(.*)$') {
        [Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim(), 'Process')
      }
    }

Esse comando não imprime os valores.

## Executar

    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    python train.py
    uvicorn app:app --reload --port 8001

Abra o chat em http://127.0.0.1:8001/assistant/ui. A página pede ASSISTANT_ACCESS_TOKEN e o mantém somente em sessionStorage na aba atual. O histórico recente do chat também fica apenas na memória da página; não é gravado no banco nem enviado para um serviço externo.

A documentação da API continua disponível em http://127.0.0.1:8001/docs.

## API do assistente

| Método | Rota | Acesso |
|---|---|---|
| GET | /assistant/ui | Interface web do chat |
| GET | /assistant/health | Público; estado de configuração |
| GET | /assistant/documents | X-Admin-Token |
| POST | /assistant/ingest | X-Admin-Token |
| POST | /assistant/chat | X-Assistant-Token |
| POST | /assistant/chat/stream | X-Assistant-Token |

Para o endpoint de streaming, envie Content-Type application/json e X-Assistant-Token. O corpo aceita question, top_k e history (até 8 turnos anteriores, cada um com role user/assistant e content). A resposta é NDJSON: começa com um evento meta, segue com eventos token e termina com done. Um evento error indica falha durante a geração.

Exemplo PowerShell para o endpoint normal:

    $body = @{
      question = 'Como começo a diagnosticar uma falha de resolução DNS?'
      top_k = 4
    } | ConvertTo-Json

    Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8001/assistant/chat' -Headers @{ 'X-Assistant-Token' = $env:ASSISTANT_ACCESS_TOKEN } -ContentType 'application/json' -Body $body

## Indexar um documento

Envie apenas documentação que você tem autorização para usar. Não envie dados pessoais, registros de clientes, senhas, tokens, códigos MFA, segredos ou incidentes sem anonimização.

    $body = @{
      source = 'network-dns-runbook.md'
      content = 'DNS: confirme usuários afetados, endereço IP, gateway e resolvedores configurados. Teste a resolução com nslookup e registre a mensagem de erro. Não altere configurações de produção sem aprovação.'
    } | ConvertTo-Json

    Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8001/assistant/ingest' -Headers @{ 'X-Admin-Token' = $env:ASSISTANT_ADMIN_TOKEN } -ContentType 'application/json' -Body $body

A resposta contém o número de trechos indexados e o modelo de embedding usado.

## Provedores opcionais

O Hugging Face continua disponível como alternativa:

    LLM_PROVIDER=huggingface
    HF_TOKEN=seu_token
    HF_CHAT_MODEL=Qwen/Qwen3.5-9B:cheapest

O token precisa da permissão Make calls to Inference Providers. A conta testada retornou HTTP 402 exigindo créditos pré-pagos ou créditos incluídos em plano PRO. Por isso, o modo Hugging Face não é o padrão e não deve ser tratado como gratuito para essa conta.

Para usar OpenAI em uma instalação com créditos, defina LLM_PROVIDER=openai, RAG_EMBEDDING_PROVIDER=openai e configure OPENAI_API_KEY e os nomes de modelos.

Ao mudar o modelo de embedding, os vetores de outro modelo não são comparáveis. Reindexe os documentos que deseja consultar com o novo modelo; os vetores antigos são mantidos sob o namespace anterior.

## Segurança e limitações

O conteúdo recuperado e o histórico do navegador são dados não confiáveis. Instruções inseridas em documentos ou no histórico não podem alterar o papel do assistente. Fatos organizacionais devem ser sustentados pelo contexto recuperado. O assistente não solicita senhas, tokens ou códigos MFA e não executa ações em sistemas.

Mantenha a API em rede confiável. Antes de expor para terceiros, use HTTPS, controle de identidade e proteção contra abuso. O token da interface é uma barreira simples para uso local, não substitui um sistema de autenticação multiusuário.

- Busca linear em SQLite; avalie pgvector ou banco vetorial dedicado para escalar.
- Chunking baseado em caracteres, não tokens.
- O limiar RAG_MIN_RELEVANCE exige avaliação com perguntas reais.
- O histórico é limitado aos últimos oito turnos e existe apenas na sessão atual.
- Faltam avaliações sistemáticas de precisão/recall, qualidade de citações, abstenção e latência. Não classifique o módulo como pronto para produção sem essas avaliações.
