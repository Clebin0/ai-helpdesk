# Modelo

## O que ele faz

O classificador recebe o texto de um chamado e tenta encaixar em uma das categorias:

- network
- hardware
- software
- access
- printing
- security
- other

O pipeline usa TF-IDF com n-grams de 1 e 2 termos. O classificador escolhido no treino é salvo e carregado pela API em FastAPI.

## Dados

O dataset deste repositório foi montado só para teste do fluxo. Não tem chamado real, dado de usuário, credencial ou log de empresa.

## Como estou avaliando

No treino eu comparo os modelos por Macro F1 usando validação cruzada. Depois avalio o escolhido em um holdout separado.

Também salvo:

- accuracy
- macro F1
- weighted F1
- F1 por categoria
- matriz de confusão

Esses números servem para comparar versões dentro deste laboratório. Não são uma estimativa de desempenho em produção.

## Limites atuais

Hoje o projeto ainda não trata bem coisas como mudança de vocabulário, categorias desbalanceadas, calibração de confiança ou histórico real de chamados.

Chamados de segurança, incidentes críticos e previsões com confiança baixa continuam marcados para revisão manual.
