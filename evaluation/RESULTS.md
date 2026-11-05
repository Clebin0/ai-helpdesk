# RAG benchmark — resultado de referência

**Data:** 5 de novembro de 2026  
**Corpus:** 8 documentos sintéticos e 8 perguntas em português  
**Embedding:** FastEmbed — sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  
**Retriever:** SQLite + similaridade por cosseno  
**K:** 4 fontes únicas  
**Repetições de latência:** 3 por pergunta

## Comparação do limite de relevância

| Métrica | RAG_MIN_RELEVANCE=0.18 | RAG_MIN_RELEVANCE=0.32 |
|---|---:|---:|
| Recall@4 | 100% | 100% |
| Precision@4 (denominador fixo) | 31,25% | 31,25% |
| Precision of returned | 31,25% | 60,42% |
| Fontes retornadas por pergunta (média) | 4,00 | 2,38 |
| MRR | 100% | 100% |
| Hit Rate@4 | 100% | 100% |
| Latência média | 7,451 ms | 9,474 ms |
| Latência p95 | 9,274 ms | 11,458 ms |

Com 0.32, o retriever manteve todas as fontes esperadas entre as quatro primeiras e reduziu a média de fontes fornecidas ao contexto de 4 para 2,38. A precisão entre as fontes efetivamente retornadas aumentou de 31,25% para 60,42%. A latência medida aumentou aproximadamente 2 ms.

**Interpretação:** 0.32 é um candidato de experimentação para este corpus sintético, não um novo padrão de produção. O arquivo .env e o valor padrão do projeto não foram alterados. Antes de mudar a configuração real, amplie o conjunto de perguntas, inclua paráfrases e consultas sem resposta na base, e valide com documentação representativa e autorizada.

## Auditoria de citações de respostas reais

Foram geradas três respostas pelo Qwen 3.5 local (qwen3.5:4b-q4_K_M) para perguntas sintéticas de DNS, conectividade e phishing, com o limite 0.32.

| Métrica | Resultado |
|---|---:|
| Integridade dos IDs de citação | 100% |
| Precisão de citações alinhadas às fontes esperadas | 100% |
| Recall de fontes esperadas citadas | 100% |
| Cobertura de citações relevantes | 100% |

A auditoria verifica se [S1], [S2] etc. existem nos metadados e apontam para as fontes anotadas como relevantes. **Não mede entailment semântico**: ainda é necessária uma revisão humana para confirmar que cada trecho realmente sustenta a afirmação correspondente.

## Limitações

- O corpus contém apenas oito documentos e oito perguntas escritas para o próprio benchmark.
- Os valores não representam desempenho esperado em uma central de atendimento real.
- Perguntas reais, paráfrases, erros de digitação, consultas ambíguas e cenários sem documentação ainda precisam ser incluídos.
- Não foram avaliados faithfulness/entailment, precisão factual de cada passo, abstenção em casos sem resposta ou qualidade operacional por analistas.
- Os resultados de latência são de uma execução local e variam com CPU, cache e carga da máquina.

Reproduza o experimento com:

~~~powershell
python scripts/evaluate_rag.py --top-k 4 --repeats 3
python scripts/evaluate_rag.py --top-k 4 --repeats 3 --min-relevance 0.32
~~~

Os relatórios completos gerados em evaluation/results/ são arquivos locais ignorados pelo Git. O banco SQLite do benchmark também fica separado da base de conhecimento do assistente.
