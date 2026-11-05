# RAG benchmark — resultado de referência

**Data:** 5 de novembro de 2026  
**Dataset:** versão 2 — 8 documentos sintéticos, 8 perguntas canônicas, 8 paráfrases e 5 consultas sem resposta prevista  
**Embedding:** FastEmbed — sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  
**Retriever:** SQLite + similaridade por cosseno  
**K:** 4 fontes únicas  
**Repetições de latência:** 3 por pergunta  
**Chamadas ao LLM:** 0

## Resultado principal — versão 2

As métricas de recuperação consideram apenas as 16 perguntas respondíveis. A avaliação de abstenção considera as 21 perguntas e trata uma lista vazia após o filtro de relevância como abstenção do retriever.

| Métrica | Threshold 0.18 | Threshold 0.32 |
|---|---:|---:|
| Recall@4 (16 respondíveis) | 100% | 93,75% |
| Precision@4 (denominador fixo) | 31,25% | 29,69% |
| Precision of returned | 38,02% | 58,85% |
| Fontes retornadas por pergunta (média) | 3,62 | 2,19 |
| MRR | 100% | 93,75% |
| Hit Rate@4 | 100% | 93,75% |
| Latência média | 10,261 ms | 9,248 ms |
| Latência p95 | 12,569 ms | 12,365 ms |
| Abstenções corretas | 0/5 | 1/5 |
| Respostas indevidas em consultas sem fonte | 5/5 | 4/5 |
| Perguntas respondíveis sem recuperação | 0 | 1 |
| F1 de abstenção | 0% | 28,57% |

### Canônicas versus paráfrases

| Tipo | Threshold | Recall@4 | Precision of returned | Fontes médias |
|---|---:|---:|---:|---:|
| Canônicas (8) | 0.18 | 100% | 31,25% | 4,00 |
| Canônicas (8) | 0.32 | 100% | 60,42% | 2,38 |
| Paráfrases (8) | 0.18 | 100% | 44,79% | 3,25 |
| Paráfrases (8) | 0.32 | 87,5% | 57,29% | 2,00 |

O threshold 0.32 reduz o contexto e melhora a precisão entre as fontes retornadas, mas perde uma paráfrase válida e continua trazendo fontes para quatro das cinco perguntas não cobertas. Portanto, não é seguro promovê-lo ao ambiente de produção.

## Varredura experimental de thresholds

As cinco execuções abaixo usaram uma repetição por pergunta; elas servem para comparar recuperação e abstenção, não para comparar latência.

| Threshold | Recall@4 | Precision of returned | Abstenções corretas | Falsas recuperações sem resposta | Respondíveis perdidas | F1 de abstenção |
|---|---:|---:|---:|---:|---:|---:|
| 0.35 | 90,6% | 65,1% | 1/5 | 4/5 | 1 | 28,6% |
| 0.40 | 90,6% | 81,3% | 2/5 | 3/5 | 1 | 50,0% |
| 0.45 | 78,1% | 84,4% | 2/5 | 3/5 | 2 | 44,4% |
| 0.50 | 62,5% | 75,0% | 4/5 | 1/5 | 4 | 61,5% |
| 0.55 | 53,1% | 62,5% | 5/5 | 0/5 | 6 | 62,5% |

Mesmo quando 0.55 evita todas as cinco falsas recuperações, perde seis consultas respondíveis. Aumentar apenas o threshold não oferece um ponto de operação equilibrado neste corpus. Isso motivou o experimento de reranking lexical sobre candidatos vetoriais, medido abaixo.

## Reranking lexical + vetorial — experimento v2

O modo experimental `--retrieval-mode hybrid` parte dos oito candidatos mais próximos pela busca vetorial e os reordena por uma combinação de score semântico e cobertura lexical IDF. A normalização inclui variações comuns do vocabulário de suporte (por exemplo, “trazer de volta” e “restauração”). **Não é uma busca BM25/FTS independente**: o reranker só pode reordenar candidatos que a busca densa já encontrou.

Parâmetros usados no experimento reproduzível:

| Parâmetro | Valor |
|---|---:|
| Peso semântico | 0,80 |
| Peso lexical | 0,20 |
| Cobertura lexical mínima | 0,10 |
| Resgate por score semântico | 0,55 |
| RAG_MIN_RELEVANCE da execução | 0,18 |
| K final | 4 |
| Repetições | 3 |

### Comparação com o retriever semântico

| Métrica | Semântico (0.18) | Reranker (0.18) |
|---|---:|---:|
| Recall@4 de fontes esperadas | 100% | 93,75% |
| Precision of returned | 38,02% | 91,67% |
| Fontes retornadas por pergunta | 3,62 | 1,38 |
| MRR | 100% | 100% |
| Hit Rate@4 | 100% | 100% |
| Consultas respondíveis com ao menos uma fonte esperada | 16/16 | 16/16 |
| Abstenções corretas em consultas sem fonte | 0/5 | 5/5 |
| Consultas sem fonte com recuperação indevida | 5/5 | 0/5 |
| Consultas respondíveis sem fonte esperada | 0 | 0 |
| F1 de abstenção | 0% | 100% |
| Latência média | 8,541 ms | 9,462 ms |
| Latência p95 | 11,012 ms | 12,117 ms |

### Canônicas versus paráfrases com reranking

| Tipo | Recall@4 | Precision of returned |
|---|---:|---:|
| Canônicas (8) | 93,75% | 87,50% |
| Paráfrases (8) | 93,75% | 95,83% |

Neste corpus, o reranker reduziu bastante as fontes irrelevantes enviadas ao contexto e se absteve nas cinco perguntas sem documentação, sem deixar perguntas respondíveis completamente sem evidência esperada. Em contrapartida, perdeu uma fonte esperada dentro de uma consulta que possuía mais de uma fonte correta. A latência média aumentou cerca de 0,9 ms.

**Interpretação:** o experimento é promissor para reduzir contexto irrelevante e melhorar a abstenção, mas os números de 100% são observados em apenas cinco perguntas sintéticas fora do escopo e oito documentos sintéticos. Não são evidência de segurança ou qualidade em produção. Também não avaliam se a resposta textual do LLM recusa corretamente nem se cada afirmação gerada é sustentada pelas fontes.

Reproduza com:

~~~powershell
python scripts/evaluate_rag.py --retrieval-mode hybrid --top-k 4 --repeats 3 --min-relevance 0.18
~~~

O modo padrão continua sendo `semantic`. O modo `hybrid` só é ativado explicitamente no benchmark e não foi integrado ao caminho de chat.

**Configuração de produção preservada:** `RAG_MIN_RELEVANCE=0.18`. Nenhum valor do arquivo .env ou do padrão de produção foi alterado. Os resultados são experimentais e não justificam uma alteração automática.

## Auditoria anterior de citações

Em uma execução anterior, três respostas sintéticas geradas pelo Qwen 3.5 local foram auditadas para DNS, conectividade e phishing.

| Métrica | Resultado |
|---|---:|
| Integridade dos IDs de citação | 100% |
| Precisão de citações alinhadas às fontes esperadas | 100% |
| Recall de fontes esperadas citadas | 100% |
| Cobertura de citações relevantes | 100% |

A auditoria verifica se [S1], [S2] etc. existem nos metadados e apontam para as fontes anotadas como relevantes. **Não mede entailment semântico** e não confirma se cada trecho sustenta todas as afirmações da resposta. Além disso, a avaliação v2 deste relatório mede abstenção no retriever, não a qualidade textual de uma recusa do LLM.

## Limitações

- O corpus tem somente oito documentos sintéticos e 21 perguntas escritas para o benchmark; cinco consultas sem resposta são uma amostra muito pequena.
- Os números não representam desempenho esperado em uma central de atendimento real.
- Ainda faltam consultas ambíguas, erros de digitação, documentos extensos e documentação real autorizada e anonimizada.
- As métricas não avaliam faithfulness/entailment, veracidade de cada afirmação, nem a qualidade de recusa gerada pelo LLM.
- Os resultados de latência são locais e variam com CPU, cache e carga da máquina.

Reproduza a avaliação principal com:

~~~powershell
python scripts/evaluate_rag.py --top-k 4 --repeats 3
python scripts/evaluate_rag.py --top-k 4 --repeats 3 --min-relevance 0.32
~~~

Para uma varredura adicional, execute o mesmo comando com `--min-relevance` em valores como 0.35, 0.40, 0.45, 0.50 e 0.55. O valor é aplicado apenas ao processo atual.

Os relatórios JSON gerados em `evaluation/results/` são arquivos locais ignorados pelo Git. O banco SQLite do benchmark também fica separado da base de conhecimento do assistente.
