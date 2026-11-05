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

Mesmo quando 0.55 evita todas as cinco falsas recuperações, perde seis consultas respondíveis. Aumentar apenas o threshold não oferece um ponto de operação equilibrado neste corpus. A próxima hipótese técnica é comparar recuperação híbrida (lexical + vetorial) e reranking antes de calibrar novamente a abstenção.

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
