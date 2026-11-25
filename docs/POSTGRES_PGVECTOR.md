# PostgreSQL + pgvector com Ollama local

Esta configuração usa PostgreSQL com pgvector para persistir e pesquisar os vetores do módulo RAG. A geração de respostas continua local via Ollama e os embeddings continuam locais via FastEmbed. Com essa configuração, documentos, perguntas e respostas não precisam ser enviados a uma API de IA hospedada.

## Arquitetura

- Geração: Ollama executando o modelo Qwen localmente.
- Embeddings: FastEmbed com sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384 dimensões).
- Persistência: PostgreSQL com a extensão pgvector.
- Busca vetorial: distância cosseno, com índice HNSW.
- API: FastAPI, mantendo os endpoints existentes de triagem e assistente.

O PostgreSQL é selecionado quando DATABASE_URL contém uma URL postgresql:// ou postgres://. Sem essa variável, a aplicação mantém o backend SQLite anterior para compatibilidade e testes. Não há migração automática dos dados SQLite existentes.

## Requisitos

- Windows 10/11 ou Linux/macOS.
- Docker Desktop com Docker Compose V2.
- Memória e armazenamento suficientes para o modelo Ollama escolhido. Execução em CPU é suportada, mas pode ser lenta.

## Configuração no Windows (PowerShell)

Na raiz do repositório:

~~~powershell
Copy-Item .env.example .env
~~~

Gere dois tokens independentes:

~~~powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
python -c "import secrets; print(secrets.token_urlsafe(32))"
~~~

Abra o arquivo .env e preencha ASSISTANT_ACCESS_TOKEN e ASSISTANT_ADMIN_TOKEN com os valores gerados. Não reutilize senhas pessoais e não envie o arquivo .env ao GitHub.

Os valores de PostgreSQL no exemplo são destinados somente ao ambiente local. Altere POSTGRES_PASSWORD antes de usar qualquer servidor compartilhado. O Compose define o hostname interno do banco como postgres; o .env contém a URL localhost para execução da API fora do Docker.

## Subir os serviços

Inicie o PostgreSQL e o Ollama:

~~~powershell
docker compose up -d postgres ollama
~~~

Baixe o modelo de chat no volume persistente do Ollama:

~~~powershell
docker compose exec ollama ollama pull qwen3.5:4b-q4_K_M
~~~

Construa e inicie a API:

~~~powershell
docker compose up -d --build app
docker compose logs -f app
~~~

O contêiner da API treina/carrega o classificador existente antes de iniciar o Uvicorn. A migração 001_pgvector é aplicada automaticamente na primeira operação de armazenamento PostgreSQL. O primeiro uso do FastEmbed baixa o modelo de embeddings; os arquivos ficam no volume persistente fastembed_cache.

Acesse:
- Interface do assistente: http://localhost:8000/assistant/ui
- Documentação interativa: http://localhost:8000/docs
- Saúde do assistente: http://localhost:8000/assistant/health
- Saúde geral: http://localhost:8000/health

## Validar o PostgreSQL

~~~powershell
docker compose exec postgres psql -U helpdesk -d ai_helpdesk -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
docker compose exec postgres psql -U helpdesk -d ai_helpdesk -c "SELECT version FROM schema_migrations ORDER BY version;"
docker compose exec postgres psql -U helpdesk -d ai_helpdesk -c "\d rag_chunks"
~~~

A coluna embedding é vector(384). O código valida a dimensão e rejeita valores não finitos antes de gravar. O índice HNSW usa vector_cosine_ops. Se o modelo de embeddings for trocado por outro com dimensão diferente, será necessária uma migração explícita do esquema e reindexação dos documentos.

## Testar ingestão e consulta

No PowerShell, use o token administrativo para indexar documentação de teste autorizada:

~~~powershell
$admin = "COLE_AQUI_O_ASSISTANT_ADMIN_TOKEN"
$chat = "COLE_AQUI_O_ASSISTANT_ACCESS_TOKEN"

$ingest = @{
  source = "runbook-dns-demo.md"
  content = "Diagnóstico DNS: confirme o endereço IP, o gateway e os resolvedores configurados. Execute nslookup para validar a resolução do domínio. Registre o resultado antes de alterar configurações."
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://localhost:8000/assistant/ingest" -Headers @{ "X-Admin-Token" = $admin } -ContentType "application/json" -Body $ingest
~~~

Consulte os documentos indexados:

~~~powershell
Invoke-RestMethod -Method Get -Uri "http://localhost:8000/assistant/documents" -Headers @{ "X-Admin-Token" = $admin }
~~~

Faça uma pergunta:

~~~powershell
$question = @{
  question = "Como posso validar se o problema é de resolução DNS?"
  top_k = 4
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://localhost:8000/assistant/chat" -Headers @{ "X-Assistant-Token" = $chat } -ContentType "application/json" -Body $question
~~~

A resposta inclui as fontes recuperadas. Quando não houver evidência relevante, o sistema deve se abster em vez de inventar um procedimento. A geração só ocorre depois que a recuperação encontra trechos acima do limiar configurado.

## Testes automatizados

Os testes unitários mantêm SQLite temporário para serem rápidos e isolados. O teste de integração PostgreSQL grava, consulta, lista e remove um documento sintético quando RUN_POSTGRES_TESTS=1 e DATABASE_URL estão definidos:

~~~powershell
python -m pip install -r requirements.txt
$env:DATABASE_URL = "postgresql://helpdesk:local_dev_change_me@localhost:5432/ai_helpdesk"
$env:RUN_POSTGRES_TESTS = "1"
python -m pytest -q
~~~

O workflow do GitHub Actions também inicia um serviço PostgreSQL com pgvector e executa esse teste junto com os testes SQLite.

## Limitações conhecidas

- O teste de integração valida persistência, recuperação vetorial e metadados; não mede a qualidade das respostas do LLM.
- O modelo de embeddings atual está fixado em 384 dimensões no primeiro esquema.
- O HNSW é útil para busca aproximada em conjuntos maiores; o conjunto sintético atual ainda é pequeno.
- O histórico da conversa continua limitado à sessão do navegador.
- Não existe migração automática do SQLite legado.
- Autenticação por token é adequada apenas para demonstração local; não substitui identidade, autorização multiusuário, rate limiting, HTTPS ou gestão de segredos para produção.
- Use somente documentação autorizada e anonimizada. Não indexe dados pessoais, senhas, tokens, códigos MFA ou dados reais de clientes.
