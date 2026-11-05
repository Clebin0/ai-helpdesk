# AI Helpdesk — copiloto RAG com Hugging Face

O classificador original TF-IDF continua separado e inalterado. O módulo /assistant adiciona busca semântica e geração de respostas com fontes.

## Arquitetura

1. Um mantenedor envia documentação aprovada para POST /assistant/ingest.
2. O FastEmbed gera embeddings localmente, na CPU, usando um modelo multilíngue baixado do Hugging Face Hub.
3. Os textos, as fontes e os vetores ficam armazenados no SQLite local.
4. O usuário envia uma pergunta para POST /assistant/chat.
5. O sistema gera o embedding da pergunta localmente, calcula similaridade por cosseno e descarta resultados pouco relevantes.
6. A Hugging Face Inference Providers gera uma resposta usando somente os trechos recuperados; a resposta inclui referências como [S1].

O endpoint OpenAI-compatible da Hugging Face é usado apenas para chat. Os embeddings não são enviados à API: são calculados localmente com FastEmbed/ONNX. O primeiro uso baixa o modelo; depois disso, a geração de embeddings funciona localmente e não consome créditos de inferência.

## Configuração local (PowerShell)

Copie o exemplo e edite o arquivo local:

    Copy-Item .env.example .env
    notepad .env

No arquivo .env, preencha HF_TOKEN, ASSISTANT_ACCESS_TOKEN e ASSISTANT_ADMIN_TOKEN. Depois de editar, carregue as variáveis na mesma sessão do PowerShell que vai iniciar a API ou enviar chamadas:

    Get-Content .env | ForEach-Object {
      if ($_ -match '^\s*([^#=]+)\s*=(.*)$') {
        [Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim(), 'Process')
      }
    }

Esse comando não imprime os valores dos tokens.

Para criar o token da Hugging Face:

1. Acesse https://huggingface.co/settings/tokens.
2. Crie um token Fine-grained.
3. Habilite a permissão Make calls to Inference Providers.
4. Copie o token para HF_TOKEN no .env. Não envie esse valor no chat, não o coloque no Git e não tire print mostrando o token.

O padrão usa Qwen/Qwen3.5-9B:cheapest no chat. O sufixo :cheapest pede ao roteador da Hugging Face o provedor disponível com menor custo para esse modelo. Você pode trocar HF_CHAT_MODEL por outro modelo listado em https://huggingface.co/inference/models.

## Limite de uso

Contas gratuitas da Hugging Face recebem atualmente US$ 0,10 por mês para Inference Providers; esse valor pode mudar. Depois que a cota acabar, novas chamadas podem falhar até o próximo ciclo ou até a conta adicionar créditos. Este projeto não compra créditos nem habilita recarga. Confira o uso em https://huggingface.co/settings/billing e não adicione forma de pagamento para este teste.

## Instalação e execução

    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    python train.py
    uvicorn app:app --reload --port 8001

Abra http://127.0.0.1:8001/docs.

A primeira indexação pode levar mais tempo porque o FastEmbed baixa o modelo sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2. O download é feito do Hugging Face Hub; as inferências de embeddings seguintes são locais.

## Verificar a configuração

    Invoke-RestMethod -Uri 'http://127.0.0.1:8001/assistant/health'

A resposta mostra o provedor, os modelos e indicadores booleanos de configuração. Ela nunca retorna tokens ou chaves.

## Indexar um documento

Envie apenas documentação que você tem autorização para usar. Não envie dados pessoais, registros de clientes, senhas, tokens, códigos MFA, segredos ou incidentes sem anonimização.

    $body = @{
      source = 'network-dns-runbook.md'
      content = 'DNS: confirme usuários afetados, endereço IP, gateway e resolvedores configurados. Teste a resolução de nomes e registre o erro exato. Não altere configurações de produção sem aprovação.'
    } | ConvertTo-Json

    Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8001/assistant/ingest' -Headers @{ 'X-Admin-Token' = $env:ASSISTANT_ADMIN_TOKEN } -ContentType 'application/json' -Body $body

## Fazer uma pergunta

    $body = @{ question = 'Como começo a diagnosticar uma falha de resolução DNS?'; top_k = 4 } | ConvertTo-Json

    Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8001/assistant/chat' -Headers @{ 'X-Assistant-Token' = $env:ASSISTANT_ACCESS_TOKEN } -ContentType 'application/json' -Body $body

O resultado contém a resposta, os modelos utilizados e as fontes citadas. O modelo é orientado a declarar quando a base não contém informação suficiente. Isso reduz respostas sem suporte, mas não garante correção factual: confira as fontes antes de agir.

## Provedores opcionais

O padrão é LLM_PROVIDER=huggingface e RAG_EMBEDDING_PROVIDER=local.

Para usar OpenAI em uma instalação que tenha créditos, defina LLM_PROVIDER=openai e RAG_EMBEDDING_PROVIDER=openai, configure OPENAI_API_KEY e os nomes de modelos. Não é necessário configurar OpenAI para o modo Hugging Face + embeddings locais.

Ao mudar o modelo de embedding, documentos já indexados com outro modelo não aparecem na nova coleção lógica. Reindexe os documentos que deseja consultar com o novo modelo. Os vetores antigos não são apagados automaticamente.

## Segurança e limites conhecidos

- A API exige X-Admin-Token para ingestão/listagem e X-Assistant-Token para chat.
- O conteúdo recuperado é tratado como dado não confiável; instruções inseridas em documentos não podem mudar o papel do assistente.
- O sistema não solicita senhas, tokens ou códigos MFA e não executa ações nos sistemas.
- Mantenha o serviço em rede confiável. Antes de expor para terceiros, use HTTPS, controle de identidade e proteção contra abuso.
- A busca linear em SQLite atende ao objetivo didático; avalie pgvector ou banco vetorial dedicado para escalar.
- Chunking é baseado em caracteres, não em tokens. O limite de relevância RAG_MIN_RELEVANCE precisa ser avaliado com perguntas reais e documentos revisados.
- Ainda faltam avaliações sistemáticas de precisão/recall, qualidade das citações, abstenção e latência. Não classifique o módulo como pronto para produção sem essas avaliações.

## Diagnóstico de erros

- HTTP 402: a cota/crédito de inferência da Hugging Face acabou ou não está disponível. Confira o painel; o sistema não habilita compra de créditos.
- HTTP 429: limite temporário do provedor. Aguarde antes de repetir.
- HTTP 503: configuração ausente, token inválido/sem permissão ou dependência local indisponível.
- HTTP 502: falha não classificada do provedor. Consulte os logs locais sem compartilhar chaves ou tokens.
