# Plano de estudo: Previsão de Demanda com Variáveis Externas

> Trabalho final da disciplina de Machine Learning do MBA AI Engineering & Multi-Agents (FIAP).
> Este arquivo é o guia para construir o trabalho passo a passo no VS Code com o plugin do Claude.

---

## 1. Contexto

### O trabalho
A disciplina pede uma proposta de aplicação de um modelo de Machine Learning para resolver um problema real da **Quantum Commerce**, uma empresa fictícia de varejo omnicanal presente em 12 países, com mais de 5 milhões de SKUs. Ela enfrenta o "muro da escalabilidade operacional": logística, suporte e marketing não escalam mais por métodos tradicionais.

Entre os desafios propostos no enunciado, escolhi:

> **Previsão de demanda:** construir um modelo de séries temporais capaz de antecipar picos de venda por categoria ou região, reduzindo custos logísticos.

### Regras de escopo do enunciado
- Usar apenas modelos vistos na disciplina: regressão, árvores e ensembles, clusterização, séries temporais, aprendizado por reforço e introdução a redes neurais (perceptron, MLP, descida do gradiente)
- Redes convolucionais, recorrentes (LSTM), Transformers e modelos generativos **não** podem ser a base da proposta. Só podem ser citados como evolução futura
- Comparar com um **baseline simples** e, quando fizer sentido, um **modelo clássico contra uma rede neural (MLP)**

### O que precisa ser entregue (4 tópicos)
1. **Introdução e problemática:** qual desafio e por que é relevante
2. **Motivação e justificativa do modelo:** por que esse modelo, comparação com alternativas, baseline e clássico vs MLP
3. **Proposta de solução:** modelo, dados, pipeline de treino, métricas e como o resultado é consumido na Quantum Commerce
4. **Considerações e impacto:** limitações, riscos e valor gerado para o negócio

### Critérios de avaliação
- Material entregue (documento, slides ou notebook)
- Aderência e profundidade nos tópicos
- Justificativa técnica na escolha do modelo
- Originalidade e aderência ao contexto da Quantum Commerce

---

## 2. Por que escolhi este tema

1. **Ponte com o trabalho final do MBA (TCC):** a ideia do TCC é um modelo com dados de agro (cultura, safra, produtividade, ano, quantidade produzida) que prevê se a safra vai ser **maior ou menor**, dentro de uma pipeline maior orquestrada por IA (dados de clima, imposto de renda do produtor etc.). Previsão de demanda com variáveis externas usa exatamente as mesmas técnicas.
2. **Dificuldade média:** dataset pequeno, limpo e sem dados faltantes.
3. **Bom para visualizar:** séries temporais rendem gráficos claros (real vs previsto, picos, feriados).

### O paralelo com o TCC

| Este trabalho (Walmart) | TCC (agro) |
|---|---|
| Loja | Município / região |
| Semana | Ano-safra |
| `Weekly_Sales` (alvo) | Produtividade (kg/ha) ou quantidade produzida |
| `Temperature` | Dados de clima (chuva, temperatura) |
| `Holiday_Flag` | Eventos (El Niño, geada) |
| `Fuel_Price`, `CPI` | Preço de insumos, diesel |
| `lag_52` (mesma semana do ano anterior) | Produtividade da safra anterior |
| "A próxima semana vai vender mais ou menos?" | "A safra vai ser maior ou menor?" |

### Lições que vou levar para o TCC
- **Variável externa futura também é uma previsão.** Para prever a venda da semana que vem, preciso da temperatura da semana que vem. No agro, o clima da safra futura vem de um modelo de clima. Na pipeline orquestrada, isso vira um agente de clima alimentando o modelo de safra.
- **Poucos pontos por série pedem um modelo global.** Cada loja tem só ~140 semanas, então treino um modelo com todas as lojas juntas. No agro, cada município tem poucos anos de dados, então o modelo global é praticamente obrigatório.
- **Split temporal:** treinar no passado e testar no futuro vale igual nos dois casos.

---

## 3. Sobre mim (para o plugin calibrar as explicações)

- Engenheiro de software sênior/líder técnico, +10 anos de experiência. Domino Python, testes, arquitetura e boas práticas de código. **Não precisa explicar o básico de programação.**
- **Sou novo em Machine Learning.** Conceitos de ML precisam ser explicados de forma didática, com analogias de engenharia de software (CI/CD, testes, monitoramento, capacity planning etc.).
- Aprendo fazendo: prefiro entender o porquê antes de ver o código.

---

## 4. Como quero ser guiado

Instruções para o plugin do Claude:

1. **Uma etapa por vez.** Não adiante etapas futuras.
2. **Explique antes de gerar código:** qual é o objetivo da etapa, qual conceito novo aparece e por quê.
3. **Código em blocos pequenos**, uma célula de notebook por vez, com comentários curtos.
4. **Depois de cada bloco**, diga o que devo observar no resultado (o que é esperado, o que seria sinal de problema).
5. **Ao final da etapa**, me faça a pergunta do checkpoint e espere eu responder antes de corrigir.
6. **Aponte armadilhas** (vazamento de dados, bugs silenciosos) quando aparecerem.
7. Mantenha o escopo da disciplina: nada de LSTM, Prophet, ARIMA avançado ou deep learning como solução principal.
8. Se uma etapa ficar grande demais, divida em partes "a" e "b".

---

## 5. Stack recomendada

| Ferramenta | Para que serve |
|---|---|
| **Python 3.11+** | Linguagem |
| **Jupyter Notebook (.ipynb)** | Rodar no VS Code (extensão Jupyter) ou no Google Colab |
| **pandas** | Manipular os dados (tabelas, datas, agrupamentos, lags) |
| **numpy** | Cálculos numéricos |
| **matplotlib** (e opcionalmente **seaborn**) | Gráficos |
| **scikit-learn** | Modelos, métricas, split, pipeline, normalização |
| **kagglehub** | Baixar o dataset do Kaggle (alternativa: baixar o CSV manualmente) |

Instalação local:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub jupyter
```

Organização sugerida: **um único notebook** com uma seção (título markdown) por etapa. No final, o notebook já é parte da entrega.

---

## 6. Dataset

**Walmart Dataset (Retail)** no Kaggle: https://www.kaggle.com/datasets/rutuspatel/walmart-dataset-retail

- 45 lojas Walmart
- Vendas **semanais** de 05/02/2010 a 26/10/2012 (~143 semanas por loja, ~6.435 linhas)
- Dados dos EUA (usado como proxy para validar a abordagem na Quantum Commerce)

| Coluna | O que é | Papel |
|---|---|---|
| `Store` | ID da loja (1 a 45) | Identificador da série |
| `Date` | Semana da venda (formato dia-mês-ano) | Eixo do tempo |
| `Weekly_Sales` | Vendas da loja na semana (US$) | **Alvo** |
| `Holiday_Flag` | 1 se a semana tem feriado, 0 se não | Variável externa |
| `Temperature` | Temperatura média da semana na região (°F) | Variável externa |
| `Fuel_Price` | Preço médio do combustível na região | Variável externa |
| `CPI` | Índice de preços ao consumidor (inflação) | Variável externa |
| `Unemployment` | Taxa de desemprego na região | Variável externa |

---

## 7. Modelos que vou testar

| Modelo | Tipo | Por que entra na comparação |
|---|---|---|
| **Baseline "repete a semana passada"** | Regra simples | Régua mínima. Se um modelo não vence isso, não serve |
| **Baseline "repete o ano passado"** | Regra simples | Captura sazonalidade anual (Natal, Black Friday) |
| **Regressão Linear (ou Ridge)** | Clássico, linear | Simples e explicável (um peso por feature) |
| **Random Forest** | Ensemble de árvores | Captura relações não lineares, robusto |
| **Gradient Boosting** | Ensemble de árvores | Costuma ser o mais forte em dados tabulares |
| **MLP (rede neural simples)** | Rede neural | Comparação clássico vs rede neural pedida no enunciado |
| **Regressão Logística** | Clássico, classificação | Versão "vai vender mais ou menos?" (espelho do TCC) |

### Métricas

| Métrica | O que mede | Por que usar |
|---|---|---|
| **MAE** | Erro médio absoluto, em dólares | Fácil de entender ("erramos US$ X por semana em média") |
| **WAPE** | Soma dos erros / soma das vendas reais | Erro como % do volume. Ótimo para explicar ao negócio ("erramos 7% do volume") |
| **RMSE** (opcional) | Raiz do erro quadrático médio | Pune mais os erros grandes (picos perdidos) |
| **Acurácia, precision, recall, F1** | Métricas de classificação | Só na versão "mais ou menos" |

---

## 8. Armadilhas que preciso evitar

| Armadilha | O que acontece | Como evitar |
|---|---|---|
| **Data no formato errado** | O pandas lê 05-02-2010 como 2 de maio, sem dar erro | Passar o formato explicitamente na conversão |
| **Lag sem agrupar por loja** | A primeira semana da loja 2 recebe o valor da loja 1 | Sempre criar lags com `groupby("Store")` |
| **Média móvel incluindo a semana atual** | O modelo "vê a resposta" | Deslocar uma semana antes de calcular a média |
| **Split aleatório** | O modelo treina com o futuro e a métrica fica falsamente boa | Split por data: treino no passado, teste no futuro |
| **Normalizar antes do split** | Informação do teste vaza para o treino | Usar `Pipeline` do scikit-learn |
| **Resultado bom demais** | Quase sempre é vazamento | Desconfiar e revisar antes de comemorar |

---

## 9. Etapas

Cada etapa tem: objetivo, o que fazer, conceitos novos, checkpoint e perguntas para explorar.
Uma **tabela de resultados** (DataFrame) começa na etapa 4 e ganha uma linha a cada modelo. Ela é o coração do trabalho.

---

### Etapa 0 · Setup

**Objetivo:** notebook rodando com os dados carregados.

**O que fazer**
- Criar o notebook com os títulos de todas as etapas já como seções
- Instalar as bibliotecas da stack
- Baixar o dataset (via `kagglehub` ou CSV manual)
- Carregar com pandas e mostrar as primeiras linhas

**Checkpoint:** `df.head()` mostrando as 8 colunas.

**Dica:** se a autenticação do Kaggle travar por mais de 10 minutos, baixe o CSV no site e carregue direto.

---

### Etapa 1 · Raio-X do dataset

**Objetivo:** entender formato e qualidade dos dados antes de qualquer gráfico.

**O que fazer**
- Ver tipos das colunas, nulos, duplicados e estatísticas básicas (`info`, `describe`)
- Converter `Date` para datetime com o formato dia-mês-ano explícito
- Ordenar por loja e data
- Contar: quantas lojas? quantas semanas por loja? todas têm o mesmo período?

**Conceito novo:** **dados em painel.** São várias séries temporais lado a lado, uma por loja. Como ter métricas de 45 serviços diferentes no mesmo dashboard.

**Checkpoint:** descrever o dataset em uma frase: "são X lojas, Y semanas cada, de tal data a tal data, sem nulos".

**Para explorar:** por que a conversão de data errada não dá erro, mas estraga tudo depois?

---

### Etapa 2 · Explorando o comportamento no tempo

**Objetivo:** enxergar os padrões que o modelo vai precisar aprender.

**O que fazer**
- Gráfico das vendas totais por semana (todas as lojas somadas)
- Gráfico de 3 ou 4 lojas lado a lado
- Marcar as semanas de feriado no gráfico
- Comparar o tamanho das lojas (venda média por loja, em barras)
- Scatter de vendas contra cada variável externa
- Matriz de correlação simples

**Conceitos novos**
- **Tendência:** a série sobe ou desce ao longo do tempo
- **Sazonalidade:** um padrão que se repete (pico de fim de ano)
- **Escala diferente por loja:** uma loja vende várias vezes mais que outra. Isso importa quando o modelo mistura todas

**Checkpoint:** apontar no gráfico onde estão os picos e explicar de onde vêm.

**Para explorar:** as variáveis externas parecem ter relação forte ou fraca com as vendas? Anotar essa impressão para testar na etapa 6.

**Atenção:** definir um limite de tempo. Esta etapa é divertida e vira um buraco.

---

### Etapa 3 · Lags: o passado como feature

**Objetivo:** criar as primeiras features a partir do próprio histórico, ainda sem modelo.

**O que fazer**
- Criar `lag_1` (venda da semana anterior), por loja
- Criar `lag_52` (mesma semana do ano anterior), por loja
- Criar média móvel das últimas 4 semanas, **sem incluir a semana atual**
- Medir a correlação de cada feature com a venda atual
- Contar quantas linhas ficam sem valor por causa do `lag_52`

**Conceitos novos**
- **Lag:** usar o valor passado da série como informação para prever o presente. Como usar o tráfego da semana passada para planejar a capacidade desta
- **Lag por loja:** sem agrupar, os dados de lojas diferentes se misturam (bug silencioso)
- **Vazamento de dados (data leakage):** quando o modelo tem acesso a uma informação que não teria no momento real da previsão
- **Custo do `lag_52`:** o primeiro ano de cada loja fica sem valor, e perco cerca de 1/3 dos dados. Trade-off real

**Checkpoint:** explicar por que o `lag_52` pode ser mais útil que o `lag_1` perto do Natal.

**Para explorar:** vale perder um ano de dados para ter o `lag_52`? Como seria no agro, onde cada lag é um ano inteiro?

---

### Etapa 4 · Split temporal e baselines

**Objetivo:** definir como medir e qual é a régua mínima que qualquer modelo precisa vencer.

**O que fazer**
- Separar treino e teste **por data**: as últimas ~30 semanas viram teste
- Visualizar o corte num gráfico (treino de uma cor, teste de outra)
- Implementar as funções de MAE e WAPE
- Calcular os 2 baselines no teste: "repete a semana passada" (`lag_1`) e "repete o ano passado" (`lag_52`)
- Criar a **tabela de resultados** com essas 2 linhas

**Conceitos novos**
- **Split temporal:** treino no passado, teste no futuro. Com split aleatório, o modelo treina com semanas de novembro e é testado em outubro do mesmo ano: está olhando o futuro
- **MAE e WAPE:** ver seção 7
- **Baseline:** se um modelo não vence a regra simples, ele não serve. Como um teste de performance que o serviço novo precisa passar

**Checkpoint:** tabela de resultados criada com os 2 baselines.

**Para explorar:** por que 30 semanas de teste e não 5 ou 80? O que muda?

---

### Etapa 5 · Primeiro modelo: regressão linear só com lags

**Objetivo:** treinar o primeiro modelo de verdade e comparar com os baselines.

**O que fazer**
- Separar features (X) e alvo (y) no treino e no teste
- Treinar uma Regressão Linear com `lag_1`, `lag_52` e a média móvel
- Avaliar no teste e adicionar na tabela de resultados
- Plotar real vs previsto para 2 ou 3 lojas
- Olhar os coeficientes: quanto o modelo confia em cada lag?

**Conceito novo:** **regressão linear.** O modelo aprende um peso para cada feature e soma tudo. É uma fórmula com pesos ajustados pelos dados, por isso é fácil de explicar.

**Checkpoint:** o modelo venceu os baselines? Se não venceu, isso também é resultado e vale discutir no trabalho.

**Para explorar:** onde o modelo erra mais? Em feriados? Em lojas grandes?

---

### Etapa 6 · Adicionando variáveis externas

**Objetivo:** testar se os fatores externos ajudam de verdade.

**O que fazer**
- Adicionar `Holiday_Flag`, `Temperature`, `Fuel_Price`, `CPI` e `Unemployment` ao modelo da etapa 5
- Adicionar features de calendário (semana do ano, mês)
- Comparar com e sem as variáveis externas (ablação)
- Registrar cada variação na tabela de resultados

**Conceitos novos**
- **Ablação:** tirar ou adicionar um grupo de features e medir o impacto. Como um teste A/B de features. Só fica o que provou que ajuda
- **A pegadinha do futuro:** na vida real, a temperatura da próxima semana é uma previsão, não um dado conhecido. No TCC, o clima da safra futura vem de um modelo de clima (o agente de clima da pipeline)

**Checkpoint:** responder com número: "as variáveis externas melhoraram (ou não) o WAPE em X pontos".

**Para explorar:** a impressão da etapa 2 se confirmou? Qual variável externa mais ajudou?

---

### Etapa 7 · Árvores e ensembles

**Objetivo:** testar modelos que capturam relações não lineares.

**O que fazer**
- Treinar Random Forest e Gradient Boosting com as melhores features até aqui
- Incluir a loja como feature
- Avaliar e adicionar na tabela de resultados
- Plotar a importância das features de cada modelo

**Conceitos novos**
- **Árvore de decisão:** uma sequência de `if/else` aprendida dos dados ("se é semana de feriado e lag_52 > X, então...")
- **Ensemble:** várias árvores combinadas. Random Forest treina árvores independentes e tira a média. Gradient Boosting treina árvores em sequência, cada uma corrigindo o erro da anterior
- **Modelo global:** um modelo para as 45 lojas em vez de 45 modelos. No agro isso é quase obrigatório

**Checkpoint:** tabela com 5 ou 6 linhas e um "campeão provisório".

**Para explorar:** a importância das features mudou em relação aos coeficientes da regressão linear? Por quê?

---

### Etapa 8 · Rede neural simples (MLP)

**Objetivo:** cumprir a comparação clássico vs rede neural pedida no enunciado.

**O que fazer**
- Montar um `Pipeline` do scikit-learn com `StandardScaler` + `MLPRegressor`
- Testar no máximo 3 configurações de camadas (ex: uma camada pequena, uma média, duas camadas)
- Avaliar e adicionar na tabela de resultados

**Conceitos novos**
- **Por que normalizar:** se uma feature vai de 0 a 1 e outra de 0 a 3 milhões, a descida do gradiente fica desequilibrada. Árvores não precisam disso, MLP precisa
- **Pipeline:** a normalização aprende média e desvio **só no treino**. Normalizar tudo antes do split vaza informação do futuro
- **MLP nem sempre ganha:** com poucos dados tabulares, árvores costumam vencer. Se acontecer, é um ótimo ponto de discussão

**Checkpoint:** argumentar se o ganho (ou a perda) do MLP compensa a perda de explicabilidade.

**Atenção:** não passar mais que ~50 minutos ajustando o MLP.

---

### Etapa 9 · Validação mais robusta e comparação final

**Objetivo:** garantir que o campeão não ganhou por sorte do período de teste.

**O que fazer**
- Usar `TimeSeriesSplit` (validação em janelas que avançam no tempo)
- Rodar os 3 ou 4 melhores modelos nessa validação
- Montar a tabela final com média e variação do erro
- Analisar o erro por loja e em semanas de feriado vs normais

**Conceito novo:** **walk-forward validation.** Treina até a semana X, testa as próximas, avança e repete. Como rodar o teste em vários ambientes em vez de um só.

**Checkpoint:** tabela final de comparação pronta para o documento.

**Para explorar:** existe alguma loja em que o modelo é sistematicamente ruim? O que ela tem de diferente?

**Se faltar tempo:** manter só o split da etapa 4 e citar esta validação como melhoria futura.

---

### Etapa 10 · Versão "vai vender mais ou menos?" (espelho do TCC)

**Objetivo:** transformar o problema em classificação, como o "safra maior ou menor".

**O que fazer**
- Criar o alvo binário: 1 se a venda da semana for maior que a média das 4 anteriores, 0 se não
- Ver a proporção de cada classe
- Baseline: chutar sempre a classe mais comum
- Treinar Regressão Logística e o melhor ensemble da etapa 7 (versão classificador)
- Avaliar com acurácia, precision, recall, F1 e matriz de confusão

**Conceitos novos**
- **Regressão vs classificação:** prever "quanto" vs prever "qual lado". Às vezes o negócio só precisa do lado (reforçar estoque: sim ou não)
- **Probabilidade e threshold:** o modelo diz "70% de chance de subir" e o negócio decide a partir de quanto age

**Checkpoint:** dizer qual formulação é mais útil para a logística da Quantum Commerce e por quê.

**Para explorar:** no agro, faz mais sentido prever a produtividade (número) ou só "maior ou menor"?

**Se faltar tempo:** fazer depois da entrega, como estudo para o TCC. Não é exigido pelo enunciado.

---

### Etapa 11 · Interpretação e tradução para negócio

**Objetivo:** transformar números em argumentos para o board.

**O que fazer**
- Calcular a importância por permutação no modelo final (mais confiável que a importância nativa das árvores)
- Converter o erro em impacto: "WAPE de X% significa errar Y dólares por semana por loja"
- Descrever como a logística da Quantum Commerce consumiria a previsão (ex: reforçar estoque quando o previsto passa de um limite)
- Listar as limitações (dados dos EUA, só 2,7 anos, variáveis externas futuras são previsões, sem dados por produto ou categoria)

**Checkpoint:** explicar o resultado em 3 frases para alguém não técnico.

---

### Etapa 12 · Documento final

**Objetivo:** montar a entrega seguindo os 4 tópicos do enunciado.

| Tópico do enunciado | Material vem das etapas |
|---|---|
| 1. Introdução e problemática | 1, 2 |
| 2. Motivação e justificativa do modelo | 4 a 8 (baseline, linear, árvores, MLP) |
| 3. Proposta de solução | 3, 4, 9, 10 (dados, features, pipeline, métricas, consumo) |
| 4. Considerações e impacto | 11, mais evolução futura (redes recorrentes e Transformers citados só como próximos passos) |

**Checklist da entrega**
- [ ] Problema conectado à Quantum Commerce (não só "Walmart")
- [ ] Justificativa da escolha do modelo com números da tabela de resultados
- [ ] Baseline e comparação clássico vs MLP presentes
- [ ] Split temporal explicado
- [ ] Como o resultado é consumido pelo negócio
- [ ] Limitações e riscos honestos
- [ ] Evolução futura sem usar modelos fora do escopo como base

---

## 10. Dicas finais

- Uma etapa por vez. Explicação antes do código.
- Responder o checkpoint **sozinho** antes de perguntar.
- A tabela de resultados é o fio condutor: cada etapa adiciona uma linha.
- Se uma etapa ficar grande, dividir em "a" e "b". Se ficar fácil, juntar com a próxima.
- Resultado bom demais quase sempre é vazamento.
