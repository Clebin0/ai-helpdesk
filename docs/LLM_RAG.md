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

O token precisa da permissão **Make calls to Inference Providers**. Segundo a documentação atual do Hugging Face, contas Free recebem US$ 0,10 em créditos mensais para Inference Providers; esse valor pode mudar e não representa uso ilimitado. O modelo e o provedor selecionados determinam o consumo. O sufixo `:cheapest` seleciona o provedor de menor custo disponível para o modelo.

O projeto limita a geração do Hugging Face a 220 tokens por resposta por padrão. Ajuste `HF_MAX_TOKENS` entre 64 e 2048 no arquivo `.env` para controlar o tamanho máximo de saída. Para economizar a cota experimental, mantenha respostas curtas.

Se a API retornar **HTTP 402**, não há saldo utilizável para aquela chamada ou o crédito disponível foi insuficiente. Confira o saldo e o uso na página de cobrança do Hugging Face antes de repetir a solicitação. A permissão do token autentica a chamada, mas não concede créditos extras. Consulte os valores atuais em https://huggingface.co/docs/inference-providers/pricing.

O modo Hugging Face permanece opcional. Para desenvolvimento contínuo sem cobrança por chamada, mantenha `LLM_PROVIDER=ollama` e execute o modelo localmente.

Para usar OpenAI em uma instalação com créditos, defina LLM_PROVIDER=openai, RAG_EMBEDDING_PROVIDER=openai e configure OPENAI_API_KEY e os nomes de modelos.

Ao mudar o modelo de embedding, os vetores de outro modelo não são comparáveis. Reindexe os documentos que deseja consultar com o novo modelo; os vetores antigos são mantidos sob o namespace anterior.

## Avaliação do RAG (benchmark local)

O repositório inclui um benchmark reproduzível com **8 documentos sintéticos e 21 perguntas em português**: 8 canônicas, 8 paráfrases e 5 consultas sem resposta prevista. Ele mede a recuperação feita pelo embedding local real, sem chamar o modelo de chat e sem consumir créditos de inferência.

Execute na raiz do repositório, com o ambiente virtual ativado:

    python scripts/evaluate_rag.py --top-k 4 --repeats 3

Para testar outro corte sem alterar o arquivo .env:

    python scripts/evaluate_rag.py --top-k 4 --repeats 3 --min-relevance 0.32

Opcionalmente, salve o relatório JSON:

    python scripts/evaluate_rag.py --top-k 4 --repeats 3 --output evaluation/results/latest.json

O benchmark usa por padrão `data/rag-benchmark.sqlite3`, separado de `data/assistant.sqlite3`. Assim, os documentos sintéticos não são misturados à base de conhecimento do copiloto. O primeiro uso do FastEmbed pode baixar o modelo de embeddings; depois, a indexação e as consultas são executadas localmente. O Ollama não precisa estar ativo para medir a recuperação.

### Métricas

- **Recall@K:** proporção das fontes esperadas encontradas entre as K primeiras fontes únicas.
- **Precision@K:** proporção das K posições ocupadas por fontes esperadas; posições sem resultado também contam no denominador.
- **Precision of returned:** proporção de fontes esperadas entre as fontes efetivamente retornadas após o filtro de relevância.
- **Average sources returned:** quantidade média de fontes fornecidas ao contexto por pergunta.
- **MRR:** posição da primeira fonte esperada; quanto mais cedo ela aparece, maior a pontuação.
- **Hit Rate@K:** proporção de perguntas com pelo menos uma fonte esperada recuperada.
- **Latência:** média, mediana e p95 das consultas, após a indexação. A primeira inicialização/download do modelo não entra no tempo de consulta.\n- **Answerability accuracy:** acurácia da decisão do retriever entre retornar fontes e se abster, usando o rótulo answerable do dataset.\n- **Abstention precision/recall/F1:** mede se o retriever ficou sem resultados justamente nas consultas sem resposta prevista.\n- **Answerable retrieval coverage:** proporção das perguntas respondíveis que recuperam pelo menos uma fonte esperada.\n- **Evidence precision among answered queries:** proporção das consultas respondidas que recuperam ao menos uma fonte esperada.

Os resultados são calculados por fonte única, não por chunk: vários trechos do mesmo documento não ocupam várias posições do ranking. O relatório também registra o modelo de embedding, o número de repetições e o tempo de indexação.

### Auditoria das citações

Para avaliar respostas capturadas do assistente, prepare um JSON com `query_id`, `answer` e `sources`, seguindo o formato de `evaluation/answers.example.json`. Depois execute:

    python scripts/evaluate_rag.py --answers caminho/para/respostas.json

A auditoria mede se os identificadores [S1], [S2] etc. existem nos metadados retornados e se apontam para fontes esperadas pelo benchmark. Ela **não prova que o trecho citado sustenta semanticamente cada afirmação**. Essa verificação exige revisão humana ou um protocolo de avaliação semântica separado.

Para experimentar uma avaliação semântica auxiliar, o laboratório isolado inclui um juiz local em `examples/llm_foundations/grounding.py`. Ele divide a resposta em afirmações factuais, compara cada uma com os trechos fornecidos e classifica o suporte. A associação das citações usa uma correspondência textual exata normalizada entre a afirmação e a frase original; somente as citações presentes nessa mesma frase contam para a cobertura. IDs que o juiz sugerir sem correspondência ficam visíveis como não verificados. Isso é intencionalmente conservador: paráfrases podem ficar sem associação, mas o sistema não tenta adivinhar a qual frase a citação pertence. Execute `python -m examples.llm_foundations.grounding_evaluation --input evaluation/grounding.example.json`. O juiz não participa do endpoint de produção, não usa conhecimento externo como evidência e não prova que uma afirmação é verdadeira. Como todo avaliador baseado em LLM, pode falhar; valide uma amostra manualmente e nunca trate a pontuação como certificação.

### Avaliação end-to-end: recuperação + resposta + fundamentação

O runner `scripts/evaluate_rag_e2e.py` executa o fluxo completo em uma base SQLite isolada: indexa o corpus sintético, recupera trechos, gera uma resposta com o assistente local e compara as afirmações com os trechos efetivamente enviados ao modelo. Ele também avalia consultas sem resposta prevista.

Faça primeiro uma amostra balanceada de seis perguntas (duas canônicas, duas paráfrases e duas sem resposta):

    python scripts/evaluate_rag_e2e.py

Execute as 21 perguntas para a avaliação completa:

    python scripts/evaluate_rag_e2e.py --limit 21 --output evaluation/results/e2e-latest.json

O avaliador end-to-end também permite comparar os retrievers sem alterar o endpoint de produção. O modo padrão `semantic` reproduz o fluxo atual. Para testar o reranker híbrido ou a fusão RRF com o mesmo Qwen, execute:

    python scripts/evaluate_rag_e2e.py --retrieval-mode hybrid --limit 21 --output evaluation/results/e2e-hybrid.json
    python scripts/evaluate_rag_e2e.py --retrieval-mode rrf --limit 21 --output evaluation/results/e2e-rrf.json

Os modos `hybrid` e `rrf` são experimentais e existem apenas no runner de avaliação. Eles usam os mesmos parâmetros do benchmark de recuperação: peso semântico 0.80, cobertura lexical mínima 0.10, resgate semântico 0.55 e, no RRF, corte BM25 -2.5. Os valores podem ser ajustados por CLI; o relatório registra modo e parâmetros para permitir comparação reproduzível. O modo RRF cria um índice FTS5 em memória e separado da base do assistente. Compare cobertura, abstinência, suporte das afirmações e relevância das citações antes de considerar qualquer mudança no caminho de produção.

O runner força `LLM_PROVIDER=ollama` e `RAG_EMBEDDING_PROVIDER=local`, independentemente da configuração de provedores no `.env`. Portanto, não usa créditos de APIs hospedadas. O Ollama precisa estar em execução e o modelo de chat disponível; a primeira indexação pode baixar o modelo FastEmbed. O SQLite padrão é `data/rag-e2e-benchmark.sqlite3`, separado tanto da base do assistente quanto do banco do benchmark de recuperação.

O relatório separa quatro grupos: (1) cobertura da recuperação e abstenção do retriever; (2) quantidade de respostas geradas e proxy de abstenção em perguntas sem resposta; (3) proporção de afirmações `supported`, `unsupported` e `insufficient_evidence`; (4) integridade das citações e alinhamento com as fontes esperadas. O campo `unanswerable_safe_abstention_proxy_rate` conta respostas sem afirmações factuais identificadas, mas isso **não prova** que a recusa em linguagem natural esteja correta. O juiz continua sendo um LLM: seus vereditos exigem revisão humana.

O juiz tenta gerar JSON estruturado com até duas tentativas e aumenta o limite de tokens na segunda. IDs vazios, malformados ou duplicados em campos opcionais são descartados e registrados em `limitations`; IDs sintaticamente válidos mas ausentes das fontes (por exemplo, `S99`) continuam visíveis e são contabilizados como referências inválidas. Se a avaliação do juiz falhar após a repetição, o runner preserva a resposta gerada, registra `grounding_error` e continua para a próxima pergunta. Essas respostas são contabilizadas como falhas de avaliação, nunca como prova de abstenção segura.

A amostra padrão reduz o número de chamadas locais durante o primeiro teste. `--limit 21` executa todas as perguntas. `--top-k` pode variar de 1 a 4 para manter o orçamento de contexto do juiz de evidências. O relatório inclui a resposta gerada por consulta para permitir inspeção e comparação.

O valor passado em `--min-relevance` vale apenas naquela execução. Compare Recall@K, Precision of returned, as métricas de abstenção e a quantidade de fontes retornadas; um corte alto demais pode remover uma fonte necessária. No benchmark v2, o modo semântico com 0.18 retornou fontes para as cinco consultas sem resposta. Aumentar o threshold até 0.55 bloqueou todas, mas reduziu o recall para 53,1%.

Existe um reranker lexical experimental, acionado apenas por `--retrieval-mode hybrid`:

    python scripts/evaluate_rag.py --retrieval-mode hybrid --top-k 4 --repeats 3 --min-relevance 0.18

Ele reordena os oito candidatos densos por score semântico (80%) e cobertura lexical IDF (20%), exige cobertura lexical mínima de 0.10 e mantém um caminho de resgate semântico em 0.55. No corpus v2, alcançou Recall@4 de 93,75%, Precision of returned de 91,67% e abstenção correta em 5/5 consultas sem fonte, contra 0/5 no modo semântico. Também recuperou pelo menos uma fonte esperada nas 16 perguntas respondíveis. É um resultado experimental em corpus sintético de oito documentos, não uma validação para produção. Esse módulo não faz busca lexical independente/BM25 e não é usado pelo endpoint de chat.

### Busca lexical independente + Reciprocal Rank Fusion (RRF)

O benchmark também disponibiliza --retrieval-mode rrf, que cria um índice FTS5 isolado em memória a partir dos mesmos chunks e executa uma busca lexical BM25 independente da busca vetorial:

    python scripts/evaluate_rag.py --retrieval-mode rrf --top-k 4 --repeats 3 --min-relevance 0.18 --rrf-bm25-cutoff -2.5

O runner filtra candidatos BM25 com score menor ou igual ao corte configurado, filtra os candidatos vetoriais com o gate híbrido experimental e combina os rankings com RRF (constante 60). O corte BM25 padrão de -2.5 foi escolhido apenas para o corpus sintético atual; a escala depende do índice e deve ser recalibrada para cada base. O relatório registra os parâmetros e o tempo de criação do índice FTS5.

No benchmark v2, o RRF obteve Recall@4 de 93,75%, Precision of returned de 91,67%, MRR de 96,88% e 5/5 abstenções corretas. Ele foi equivalente ao reranker anterior em recall, precisão entre fontes, fontes médias e abstenção, mas perdeu um pouco de MRR. Por isso permanece alternativa experimental, não substitui o melhor resultado medido e não está integrado ao endpoint de chat.

O modo padrão do benchmark continua semantic, e RAG_MIN_RELEVANCE=0.18 permanece inalterado na aplicação.

O corpus e as perguntas são artificiais e servem para comparar alterações no retriever. Não representam chamados reais nem garantem desempenho em produção. Mantenha o mesmo corpus, modelo de embedding, K e configuração ao comparar execuções.

## Segurança e limitações

O conteúdo recuperado e o histórico do navegador são dados não confiáveis. Instruções inseridas em documentos ou no histórico não podem alterar o papel do assistente. Fatos organizacionais devem ser sustentados pelo contexto recuperado. O assistente não solicita senhas, tokens ou códigos MFA e não executa ações em sistemas.

Mantenha a API em rede confiável. Antes de expor para terceiros, use HTTPS, controle de identidade e proteção contra abuso. O token da interface é uma barreira simples para uso local, não substitui um sistema de autenticação multiusuário.

- Busca linear em SQLite; avalie pgvector ou banco vetorial dedicado para escalar.
- Chunking baseado em caracteres, não tokens.
- O limiar RAG_MIN_RELEVANCE exige avaliação com perguntas reais.
- O histórico é limitado aos últimos oito turnos e existe apenas na sessão atual.
- O benchmark é sintético e pequeno; faltam dados representativos autorizados, revisão semântica humana de citações, teste da recusa textual do LLM e validação BM25/FTS em corpus maior. Os modos hybrid e rrf são experimentais e não estão integrados ao chat. Não classifique o módulo como pronto para produção sem essas avaliações.
