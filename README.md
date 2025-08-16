# ⚕ Diabetes Age Prediction - Linear Regression :chart_with_upwards_trend:
- Algoritmo de Machine Learning utilizado para prever a idade que mulheres acima dos 21 anos provenientes da India podem desenvolver diabetes baseado nos seguintes biomarcadores:
  - Números de gravidez
  - Glicose do sangue
  - Pressão arterial
  - Largura da pele
  - Nível de insulina
  - IMC

[![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow)]()
[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)]()
[![Pandas](https://img.shields.io/badge/Pandas-blue?logo=pandas)]()
[![Numpy](https://img.shields.io/badge/Numpy-lightgrey?logo=numpy)]()
[![LinearRegression](https://img.shields.io/badge/LinearRegression-purple)]()
[![Feito por Ryan](https://img.shields.io/badge/Feito%20por-Ryan%20Brittes-blue?logo=github)]()
---

## :beginner: Tecnologias utilizadas:
- **Pandas**
- **Numpy**
- **Matplotlib**

## Análise preliminar dos dados:
- Clique no link abaixo para ter acesso às correçaões iniciais:
- [![Análise](https://img.shields.io/badge/-Análise-orange)](./readme/graphics.md)

## :pencil: Propósito:
Com este repositório você poderá entender um pouco melhor como a técnica de Regressão Linear utilizada em Machine Learning funciona e pode ser aplicada aos problemas reais do dia a dia. Como exemplo foi utilizada uma base de dados pré-existente e real.

Para melhorar a relação entre os dados, foram utilizadas técnicas de normalização, neste algoritmo a escolhida foi normalização por dimensionamento logarítmico. No treinamento a técnica adotada foi a de gradiente descendente estocástico com minilote, que mostrou maior eficiência no treinamento, otimizando a velocidade com que o algoritmo atingiu a convergência.

O intuito da análise realizada foi tentar prever com que idade uma pessoa pode desenvolver diabetes baseada em seus biomarcadores, este estudo foi dirigido na India com mulheres acima de 21 anos. A fim de credibilizar os pesquisadores deixarei ao final da explicação o link para acesso ao arquivo original.

Caso você já tenha algum conhecimento prévio do que se trata o conceito base, sugiro que pule para a etapa de explicação do código e dos dados encontrados.

- [![Explicação](https://img.shields.io/badge/-Explicação-yellow)](./readme/explainLinearRegression.md)

## :rocket: Código:
Com o fluxo do código abaixo:
```
DiabetesAgePrediction
  └──App
  │   └─linearRegression.py
  │   └─loadData.py
  │   └─lossModel.py
  │   └─main.py
  │   └─normalizeModel.py
  │   └─plotGraphic.py
  └──files
  │    └─diabetes.csv
  └──test
  │   └─test.py
  └──requirements.txt
```
### [![App](https://img.shields.io/badge/-App-yellow)](./App)
- [![linearRegression.py](https://img.shields.io/badge/-linearRegression.py-darkgreen)](./App/linearRegression.py): Algoritmo de treinamento, aqui será realizado todo o treinamento do modelo.
- [![loadData.py](https://img.shields.io/badge/-loadData.py-darkgreen)](./App/loadData.py): Carregamento dos dados que serão utilizados pela classe LinearRegression, onde será realizada o embaralhamento das amostras e normalização para melhor compatibilidade entre os dados.
- [![lossModel.py](https://img.shields.io/badge/-lossModel.py-darkgreen)](./App/lossModel.py): Local onde as equações de perda do modelo se encontram.
- [![main.py](https://img.shields.io/badge/-main.py-darkgreen)](./App/main.py): Classe que irá rodar o treinamento e retornar os pesos, viés e perda encontradas.
- [![normalizeModelo.py](https://img.shields.io/badge/-normalizeModelo.py-darkgreen)](./App/normalizeModelo.py): Aqui ocorre a normalização que irá converter os valores atuais do dataset para valores que sejam menos divergentes, melhorando o treinamento do modelo.
- [![plotGraphic.py](https://img.shields.io/badge/-plotGraphic.py-darkgreen)](./App/plotGraphic.py): A visualização de como os dados se relacionam entre si e o gráfico de perda serão calculados nesta classe.
### [![files](https://img.shields.io/badge/-files-yellow)](./files)
- [![diabetes.csv](https://img.shields.io/badge/-diabetes.csv-darkgreen)](./files/diabetes.csv): Arquivo utilizado como fonte de dados de treinamento do modelo.
### [![test](https://img.shields.io/badge/-test-yellow)](./test)
- test.py: Algoritmo que irá rodar o código para que sua saída seja analisada. Por padrão, o método irá gerar um gráfico de Loss vs Epochs, onde será realizada uma análise de como a perca decai ou aumenta com o tempo de treinamento com as amostras.
### [![requirements.txt](https://img.shields.io/badge/-requirements.txt-yellow)](./requirements.txt)
- Documento com todas as bibliotecas utilizadas no código, para adicioná-las com facilidade dê o comando:
```
pip install -r requirements.txt
```
## :computer: Implementação Prática:
Para implementar o código e realizar seus testes localmente, clone o repositório em um pasta com:
```
git clone https://github.com/RyanBrittes/DiabetesAgePrediction.git
```
Entre no diretório em que salvou o repositório:
```
cd local_salvo
cd DiabetesAgePrediction
cd test
```
Estando na pasta **test**:
```
python test.py
```

## Resultados encontrados:
Realizando as devidas ponderações nos parâmetros para encontrar a convergência do algoritmo, foram encontrados os seguintes valores:
- Taxa de aprendizado (self.lr): 0.0001
- Eras (self.epochs): 8000
- Tamanho do lote (self.batchSize): 30
- Taxa de amostras teste (self.rateTest): 0.1
- Taxa de amostras de treinamento: 0.9
- Perca final encontrada (lossValue): 1.124837
- Viés encontrado (self.bias): 3.3708
- Pesos encontrados (self.weight)
  - Peso 'Pregnancies': 0.00681145
  - Peso 'Glucose': 0.00624196
  - Peso 'BloodPressure': 0.00426377
  - Peso 'SkinThickness': -0.00359601
  - Peso 'Insulin': -0.0013916
  - Peso 'BMI': 0.00295275
 
Gráfico que representa a perca ao longo do treinamento do algoritmo:

![Img](graphics/LossPerEpochs.png)

## Inferência com dados de teste feitos pelo modelo
- Clique no link abaixo para analisar o resultado de três amostras com seus respectivos resultados:
- [![Amostras](https://img.shields.io/badge/-Amostras-blue)](./readme/sample.md)

## Conclusão
Com o resultado encontrado do algoritmo foi possível identificar que com o conjunto de dados utilizado, existe uma certa correlação que pode nos dar um indicativo previsão baseado em biomarcadores feitos por uma análise clínica com que idade uma mulher acima dos 21 anos, proveniente da India, pode desenvolver diabetes. Porém, é importante observar que existem resultados previstos que são muito fora do real, e em um cenário médico onde números como estes são de extrema importância pois quanto antes um problema for identificado e resolvido melhor, o contrário resulta em cenários catastróficos, então há de se melhorar em muitos pontos, seja com dados mais robustos ou com a utilização de outros algoritmos.

> Este algoritmo tem apenas o intuito de analisar um conjunto de dados e mostrar os resultados encontrados ao treinar um modelo com Regressão Linear. Por tanto não é recomendado utilizar este algoritmo como base de um diagnóstico, procure um especialista na área antes de tirar qualquer conclusão.

## Documentação adicional:
Caso queira encontrar uma documentação adicional das tecnologias utilizadas, seguem os arquivos:
| Tecnologia | Doc   |
|---------------|----------------|
| Pandas   | [Pandas - Doc](https://pandas.pydata.org/docs/)   |
| Numpy | [Numpy - Doc](https://numpy.org/doc/stable/)    |
| Matplotlib | [Matplotlib - Doc](https://matplotlib.org/stable/users/index) |
| Base de dados  |  [Kanggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  |

## Considerações finais:
Estou disponível para caso hajam dúvidas ou dicas de melhorias, abaixo encontre os meios de contato comigo:
- [![LinkedIn](https://img.shields.io/badge/-LinkedIn-blue?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ryanbrittes/)
