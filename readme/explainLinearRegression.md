### :chart_with_upwards_trend: ***Regressão Linear:***
> É uma técnica estatística utilizada para encontrar a relação entre variáveis, em machine learning é possível encontrar a relação entre os atributos e rótulos de um conjunto de dados.

- Na Regressão Linear utilizamos alguns conceitos técnicos que nos auxiliam a encontrar a corelação entre os nossos dados e a melhorar análises, tornando os dados confiáveis:
  1. Equação de Regressão Linear
  2. Perda
  3. Gradiente descendente
  4. Hiperparâmetros
1. :heavy_division_sign: **Equação de Regressão Linear:**
> $y'= b + w_1 * x_1 + w_2 * x_2 + w_n * x_n ...$
  - Onde:
    - b: viés
    - w: peso
    - x: atributo de entrada
    - y': saída
  - Caso sejam dados com vários atributos, adicionamos um peso para cada tipo de atributo, como representado por $w_n * x_n$
2. :chart_with_downwards_trend: **Perda:**
> Representa uma forma de quantificarmos o quão longe o algoritmo está nas suas predições com os dados reais, ou seja, o quanto está errando.
  - _Grupo L1:_ Tipo utilizado para quantificar perdas onde no conjunto de dados temos uma relação de dados mais agrupados uns aos outros
    - *Loss L1*: $∑|actualValue-predictedValue|$
    - *Mean Absolute Error*: $(1/N) * ∑|actualValue-predictedValue|$
  - _Grupo L2:_ Tipo utilizado para quantificar perdas onde no conjunto de dados sua relação é mais espalhada ou quando queremos que nosso algoritmo esteja próximo desses valores, é uma perda mais punitiva ao quão fora da curva os resultados previstos estão dos rótulos reais.
    - *Loss L2*: $∑(actualValue-predictedValue)²$
    - *Mean Square Error*: $(1/N) * ∑(actualValue-predictedValue)²$
3. :arrow_heading_down: **Gradiente descendente:**
> Técnica iterativa que faz cálculos e ajusta o viés e os pesos até encontrar valores com perdas muito parecidas, ou seja, quando a perda não muda mais de maneira significativa com o tempo das iterações. Podemos dizer que este loop de iteração é onde o treinamento é feito.
  - Está técnica consiste em iniciar com viés e pesos com valores próximos de zero, o usuário escolhe um número de iterações e de acordo com ele o modelo irá tentar encontrar um melhor valor para os pesos e vies que irá se basear em reduzir a perda, o objetivo é reduzir ao máximo possível a perda.
  - Este conceito funciona como uma parábola, onde o seu pico máximo é denominado convergência (ponto em que o modelo não tem mudanças significativas em suas perdas). Quando o modelo chega na convergência falamos que ele convergiu e chegamos em nosso objetivo.
4. :earth_americas: **Hiperparâmetros:**
> Variáveis que nos possibilitam controlar diferentes aspectos do treinamento, temos:
  - _Taxa de aprendizado:_ velocidade com que o modelo atualiza os pesos, valor pré-definido para controlar como o modelo irá aprender. Nele existem particularidades que precisamos nos atentar como:
    - Não deixar o valor muito alto, pois se não os resultados são muito inconstantes e o modelo por consequência nunca irá convergir.
    - Não deixar o valor muito baixo, pois quando menor for maior será o tempo para o modelo convergir.
    - Por tando, é necessário realizar testes e utilizar o valor que mais se adeque ao modelo.
  - _Tamanho do Lote:_ número de amostrar que o exemplo irá processar antes de atualizar os pesos. Pode ser pouco prático utilizar apenas um exemplo por vez antes de atualizar os pesos, por tanto podemos fazer isso de maneira mais eficiente e colocar um número de exemplos por iteração, definindo assim lotes para que o modelo processe os pesos antes de atualizar. Temos duas formas de fazer esse processo:
    - Utilizando o Grandiente descendente estocástico (SGD), onde utilizamos apenas um exemplo por iteração escolhido de forma aleatória. Este resulta em resultados ruidosos.
    - Utilizando o Grandiente descendente estocástico com mini-lotes, onde utilizamos um número de amostras por iteração escolhidas de forma aleatória antes da atualização dos pesos. Os resultados apresentam menos ruido.
  - _Eras:_ É a definição de quantas vezes o modelo irá processar todos os exemplos do conjunto, então se dizermos que Eras = 1, o modelo processou todas as amostras uma vez. Quanto maior o número de Eras melhor o modelo pode ficar, porém o tempo de treinamento também será maior.