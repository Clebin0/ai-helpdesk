# Ficha do modelo

## Identificação

Classificador supervisionado de texto para triagem inicial de chamados de TI. O modelo ativo é escolhido durante o treinamento entre Logistic Regression, Complement Naive Bayes e SGD com `log_loss`.

Não é um modelo generativo. Não redige respostas, não conversa com solicitantes e não chama sistemas externos.

## Uso pretendido

Sugerir uma categoria entre `network`, `hardware`, `software`, `access`, `printing`, `security` e `other`, para apoiar uma triagem inicial em ambiente de teste.

## Dados de treinamento

- Arquivo: `data/tickets.csv`.
- Total: 70 exemplos sintéticos.
- Distribuição: dez exemplos por categoria.
- Texto: título e descrição de solicitações fictícias.
- Divisão: 70% treino e 30% holdout, estratificada.

Nenhum ticket real, nome de cliente, e-mail, credencial ou log corporativo faz parte do dataset.

## Avaliação

Os candidatos são comparados por Macro F1 médio em cinco folds dentro do conjunto de treino. O candidato com melhor média é ajustado nesse conjunto e avaliado uma vez no holdout. São registrados accuracy, Macro F1, Weighted F1, F1 por categoria e matriz de confusão.

O tamanho do conjunto é insuficiente para inferir desempenho fora deste experimento. Resultados podem mudar com a versão do scikit-learn e das demais dependências.

## Revisão humana

A API sinaliza revisão para categoria de segurança, impacto crítico, confiança abaixo de 0,72 ou margem inferior a 0,12 entre as duas primeiras classes. A prioridade vem de regras separadas do classificador.

## Fora do escopo

Uso autônomo em produção, decisão de incidentes, resposta automática ao cliente, diagnóstico de segurança e substituição da análise de um técnico. Antes de qualquer uso com dados reais, é necessário obter autorização, remover dados pessoais, rever a distribuição das classes e validar o modelo com quem atende os chamados.
