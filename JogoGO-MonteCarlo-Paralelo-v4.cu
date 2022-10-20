/* 
 Percurso na GameTree - Jogo GO (Adaptado) - utilizando algoritmo MONTE CARLO TREE SEARCH (MCTS)
 Versão Paralela (CUDA)
 Carlos Henrique Rorato Souza - Computação Paralela (2022-1)
 Variação do jogo: 
     - Tabuleiro de tamanho N*N intersecções
     - Captura de somente uma peça por vez, cercada na horizontal e vertical
     - Cálculo do score baseado na quantidade de peças pretas e brancas restantes
 É armazenado somente o estado atual de jogo, e as simulações são feitas a partir deste estado.
 Paralelização: cada bloco fará o processamento (simulações) referente à um nó filho da raiz.
*/

/* 
 Definições iniciais:
	- N é o tamanho do tabuleiro
	- qtd_jogoadas define a quantidade de jogadas (níveis da árvore)
	- num_simulacoes define a quantidade de simulações que o MCTS fará para cada nó
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#define N 19
#define qtd_jogadas N*N
#define num_simulacoes 100000

/*
 Estrutura Estado, que representa um nó da árvore, com:
	- Tabuleiro
	- Peca (qual peça foi colocada nesta jogada)
	- Score (pontuação, calculada em função específica)
	- Linha, Coluna (em qual posição essa peça foi colocada)
	- Nível (o nível da árvore no qual o nó está)
*/
struct Estado{
	char tabuleiro[N][N];
	char peca;
	int score;
	int linha;
	int coluna;
	int nivel;
};

/*
 Função que faz a geração de números pseudoaleatórios em CUDA,
 para uso na etapa de simulação.
 Para a inicialização, informamos a semente para geração dos números.
*/
__device__ void random(int* resultado, int limite) {
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	curandState_t state;
	curand_init(clock(), id, 0, &state);
	*resultado = curand(&state) % limite;
}

/*
 Função que inicializa um estado com os valores padrão, 
 sem pai e sem filhos, com tabuleiro vazio.
 Como ela é executada tanto na GPU quanto na CPU, estão disponíveis as duas versões.
*/
__device__ void inicializaEstadoPadrao(struct Estado *s){
	int i,j;
	for(i=0; i<N; i++) for(j=0; j<N; j++) s->tabuleiro[i][j] = '-';
	s->score = 0;
	s->peca = '-';
	s->linha = 0;
	s->coluna = 0;
	s->nivel = 0;
}

void inicializaEstadoPadraoHost(struct Estado *s){
	int i,j;
	for(i=0; i<N; i++) for(j=0; j<N; j++) s->tabuleiro[i][j] = '-';
	s->score = 0;
	s->peca = '-';
	s->linha = 0;
	s->coluna = 0;
	s->nivel = 0;
}

/*
 Função que calcula o score, percorrendo o tabuleiro
 do estado. Calcula-se o score do jogador que usa as
 pedras brancas. O cálculo é feito a partir da diferença
 entre as peças brancas e pretas restantes no tabuleiro.
 Como ela é executada tanto na GPU quanto na CPU, estão disponíveis as duas versões.
*/
__device__ void calculaScore(struct Estado *s){
	int i,j,p=0,b=0;
	for(i=0; i<N; i++){
		for(j=0; j<N; j++){
			if(s->tabuleiro[i][j] != '-'){
				if(s->tabuleiro[i][j] == 'p') p++;
				else b++;
			}
		
		}
	} 	
	s->score = b - p; 
}

void calculaScoreHost(struct Estado *s){
	int i,j,p=0,b=0;
	for(i=0; i<N; i++){
		for(j=0; j<N; j++){
			if(s->tabuleiro[i][j] != '-'){
				if(s->tabuleiro[i][j] == 'p') p++;
				else b++;
			}
		
		}
	} 	
	s->score = b - p; 
}

/*
 Função que insere uma peça no tabuleiro e verifica se essa inserção
 captura alguma peça da cor oposta, verificando as quatro direções
 (cima, baixo, direita e esqueda). Caso alguma peça esteja nas bordas
 do tabuleiro, ela já é considerada cercada na direção da(s) borda(s).
 A função verifica também se o movimento é suicida, isto é, se a peça 
 foi inserida numa posição onde é capturada.
 Como ela é executada tanto na GPU quanto na CPU, estão disponíveis as duas versões.
*/
__device__ void fazMovimento(struct Estado *s, char peca, int i, int j){ //peca pode ser "b" ou "p"
	int contador;
	char oponente = peca == 'b'? 'p': 'b';
	
	if(s->tabuleiro[i][j] == '-'){
		s->tabuleiro[i][j] = peca;
	
		/* olhado para cima (j-1) e verificando se a inserção da peça cercou a peça de cima */
		contador = 0;
		if(j-1>=0 && s->tabuleiro[i][j-1] == oponente){
			if (j-1 + 1 < N) {if(s->tabuleiro[i][j-1 + 1] == peca) contador++;} else contador++;
			if (j-1 - 1 >=0) {if(s->tabuleiro[i][j-1 - 1] == peca) contador++;} else contador++;
			if (i+1 < N) {if(s->tabuleiro[i + 1][j-1] == peca) contador++;} else contador++;
			if (i-1 >=0) {if(s->tabuleiro[i - 1][j-1] == peca) contador++;} else contador++;
			
			/* Se a peça está cercada, é capturada */
			if(contador == 4) s->tabuleiro[i][j-1] = '-';
		}
		
		/* olhado para baixo (j+1) e verificando se a inserção da peça cercou a peça de baixo */
		contador = 0;
		if(j+1<N && s->tabuleiro[i][j+1] == oponente){
			if (j+1 + 1 < N) {if(s->tabuleiro[i][j+1 + 1] == peca) contador++;} else contador++;
			if (j+1 - 1 >=0) {if(s->tabuleiro[i][j+1 - 1] == peca) contador++;} else contador++;
			if (i+1 < N) {if(s->tabuleiro[i + 1][j+1] == peca) contador++;} else contador++;
			if (i-1 >=0) {if(s->tabuleiro[i - 1][j+1] == peca) contador++;} else contador++;
			
			/* Se a peça está cercada, é capturada */
			if(contador == 4) s->tabuleiro[i][j+1] = '-';
		}
		
		/* olhado para a esquerda (i-1) e verificando se a inserção da peça cercou a peça da esquerda */
		contador = 0;
		if(i-1>=0 && s->tabuleiro[i-1][j] == oponente){
			if (j + 1 < N) {if(s->tabuleiro[i - 1][j + 1] == peca) contador++;} else contador++;
			if (j - 1 >=0) {if(s->tabuleiro[i - 1][j - 1] == peca) contador++;} else contador++;
			if (i - 1 + 1 < N) {if(s->tabuleiro[i -1 + 1][j] == peca) contador++;} else contador++;
			if (i - 1 - 1 >=0) {if(s->tabuleiro[i -1 - 1][j] == peca) contador++;} else contador++;
			
			/* Se a peça está cercada, é capturada */
			if(contador == 4) s->tabuleiro[i-1][j] = '-';
		}
		
		/* olhado para a direita (i+1) e verificando se a inserção da peça cercou a peça da direita */
		contador = 0;
		if(i+1<N && s->tabuleiro[i+1][j] == oponente){
			if (j + 1 < N) {if(s->tabuleiro[i + 1][j + 1] == peca) contador++;} else contador++;
			if (j - 1 >=0) {if(s->tabuleiro[i + 1][j - 1] == peca) contador++;} else contador++;
			if (i + 1 + 1 < N) {if(s->tabuleiro[i + 1 + 1][j] == peca) contador++;} else contador++;
			if (i + 1 - 1 >=0) {if(s->tabuleiro[i + 1 - 1][j] == peca) contador++;} else contador++;
			
			/* Se a peça está cercada, é capturada */
			if(contador == 4) s->tabuleiro[i+1][j] = '-';
		}
		
		/* Avaliando se a jogada é suicida, ou seja, se a peça foi inserida numa posição onde é capturada */
		contador = 0;
		if(j + 1 < N){  if(s->tabuleiro[i][j+1] == oponente) contador++; } else contador++;
		if(j - 1 >= 0){ if(s->tabuleiro[i][j-1] == oponente) contador++; } else contador++;
		if(i + 1 < N){  if(s->tabuleiro[i+1][j] == oponente) contador++; } else contador++;
		if(i - 1 >= 0){ if(s->tabuleiro[i-1][j] == oponente) contador++; } else contador++;
		
		/* Se a peça está cercada, é capturada */
		if(contador == 4) s->tabuleiro[i][j] = '-';
		
		/* Ao final, calcula o score */
		calculaScore(s);
	}
}


void fazMovimentoHost(struct Estado *s, char peca, int i, int j){ //peca pode ser "b" ou "p"
	int contador;
	char oponente = peca == 'b'? 'p': 'b';
	
	if(s->tabuleiro[i][j] == '-'){
		s->tabuleiro[i][j] = peca;
	
		/* olhado para cima (j-1) e verificando se a inserção da peça cercou a peça de cima */
		contador = 0;
		if(j-1>=0 && s->tabuleiro[i][j-1] == oponente){
			if (j-1 + 1 < N) {if(s->tabuleiro[i][j-1 + 1] == peca) contador++;} else contador++;
			if (j-1 - 1 >=0) {if(s->tabuleiro[i][j-1 - 1] == peca) contador++;} else contador++;
			if (i+1 < N) {if(s->tabuleiro[i + 1][j-1] == peca) contador++;} else contador++;
			if (i-1 >=0) {if(s->tabuleiro[i - 1][j-1] == peca) contador++;} else contador++;
			
			/* Se a peça está cercada, é capturada */
			if(contador == 4) s->tabuleiro[i][j-1] = '-';
		}
		
		/* olhado para baixo (j+1) e verificando se a inserção da peça cercou a peça de baixo */
		contador = 0;
		if(j+1<N && s->tabuleiro[i][j+1] == oponente){
			if (j+1 + 1 < N) {if(s->tabuleiro[i][j+1 + 1] == peca) contador++;} else contador++;
			if (j+1 - 1 >=0) {if(s->tabuleiro[i][j+1 - 1] == peca) contador++;} else contador++;
			if (i+1 < N) {if(s->tabuleiro[i + 1][j+1] == peca) contador++;} else contador++;
			if (i-1 >=0) {if(s->tabuleiro[i - 1][j+1] == peca) contador++;} else contador++;
			
			/* Se a peça está cercada, é capturada */
			if(contador == 4) s->tabuleiro[i][j+1] = '-';
		}
		
		/* olhado para a esquerda (i-1) e verificando se a inserção da peça cercou a peça da esquerda */
		contador = 0;
		if(i-1>=0 && s->tabuleiro[i-1][j] == oponente){
			if (j + 1 < N) {if(s->tabuleiro[i - 1][j + 1] == peca) contador++;} else contador++;
			if (j - 1 >=0) {if(s->tabuleiro[i - 1][j - 1] == peca) contador++;} else contador++;
			if (i - 1 + 1 < N) {if(s->tabuleiro[i -1 + 1][j] == peca) contador++;} else contador++;
			if (i - 1 - 1 >=0) {if(s->tabuleiro[i -1 - 1][j] == peca) contador++;} else contador++;
			
			/* Se a peça está cercada, é capturada */
			if(contador == 4) s->tabuleiro[i-1][j] = '-';
		}
		
		/* olhado para a direita (i+1) e verificando se a inserção da peça cercou a peça da direita */
		contador = 0;
		if(i+1<N && s->tabuleiro[i+1][j] == oponente){
			if (j + 1 < N) {if(s->tabuleiro[i + 1][j + 1] == peca) contador++;} else contador++;
			if (j - 1 >=0) {if(s->tabuleiro[i + 1][j - 1] == peca) contador++;} else contador++;
			if (i + 1 + 1 < N) {if(s->tabuleiro[i + 1 + 1][j] == peca) contador++;} else contador++;
			if (i + 1 - 1 >=0) {if(s->tabuleiro[i + 1 - 1][j] == peca) contador++;} else contador++;
			
			/* Se a peça está cercada, é capturada */
			if(contador == 4) s->tabuleiro[i+1][j] = '-';
		}
		
		/* Avaliando se a jogada é suicida, ou seja, se a peça foi inserida numa posição onde é capturada */
		contador = 0;
		if(j + 1 < N){  if(s->tabuleiro[i][j+1] == oponente) contador++; } else contador++;
		if(j - 1 >= 0){ if(s->tabuleiro[i][j-1] == oponente) contador++; } else contador++;
		if(i + 1 < N){  if(s->tabuleiro[i+1][j] == oponente) contador++; } else contador++;
		if(i - 1 >= 0){ if(s->tabuleiro[i-1][j] == oponente) contador++; } else contador++;
		
		/* Se a peça está cercada, é capturada */
		if(contador == 4) s->tabuleiro[i][j] = '-';
		
		/* Ao final, calcula o score */
		calculaScoreHost(s);
	}
}

/*
 Função auxiliar que copia o tabuleiro de um nó para outro.
*/
__device__ void copiarTabuleiro(struct Estado original, struct Estado *copia){
	int i,j;
	for(i=0; i<N; i++){
		for(j=0; j<N; j++){
			copia->tabuleiro[i][j] = original.tabuleiro[i][j];
		}
	}
}

/*
 Função auxiliar para o MCTS que simula o caminho até o final da árvore (fim de jogo),
 sorteando caminhos na árovore a partir do estado s (jogadas aleatórias)
 e retornando ao final o score desse jogo.
*/
__device__ void simulacaoMCTS(struct Estado s, int *resultado){
	char peca = 'p';
	int i,j;
	
	for(i = 0; i < N*N - s.nivel; i++){		
		/* Sorteia-se uma posição válida no tabuleiro e se faz a jogada */
		do{
			random(&j,N*N);
		}while(s.tabuleiro[j/N][j%N] != '-');
		fazMovimento(&s, peca,j/N,j%N);
		
		/* Inverte-se a peça a ser jogada no próximo nível */
		peca = peca == 'b'? 'p': 'b';
	}
	calculaScore(&s);
	*resultado = s.score;
}

/*
 Função MCTS (Monte Carlo Tree Search), que recebe o estado atual da árvore e
 preenche o vetor de scores, com o somatório de scores da simulação (para cada nó).
 Este algoritmo constrói somente uma árvore parcial, com um nó raiz e seus filhos.
 Está baseado na seguinte sequência: seleção da raiz e expansão dos filhos do nó raiz,
 simulação de caminhos de jogo aleatórios para cada nó que foi gerado na expansão,
 propagação do score desta configuração de jogo e a seleção da jogada (caminho
 na árvore) que trará o score mais favorável a partir das simulações. Estas duas últimas etapas
 não são feitas nessa função, mas dentro da própria função principal, a partir do vetor de scores.
 Ela é executada na GPU. Nesse contexto, a solução foi estruturada de maneira que cada bloco
 processe as operações referentes a um nó filho.
 Os filhos não são armazenados, mas a simulação indicará a melhor jogada, que será concretizada
 dentro do tabuleiro do jogo (estado atual), na função principal.
*/
__global__ void mcts(struct Estado gameTree, int *scores){
	
	/* ETAPA 1 - SELEÇÃO: seleciona o nó raiz e declara/inicializa variáveis importantes para a função */
	int i = 0;
	int k = 0;
	int stride = blockDim.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	struct Estado temp;

	/* ETAPA 2 - EXPANSÃO: faz a expansão de todos os filhos do nó selecionado (raiz) - esta é a política de expansão */
	/* Cada bloco será um filho do nó raiz, armazenado de forma temporária */
	inicializaEstadoPadrao(&temp);
	
	/* O filho continua o jogo do pai */
	copiarTabuleiro(gameTree,&temp);
	
	if(temp.tabuleiro[blockIdx.x/N][blockIdx.x%N] == '-'){
		fazMovimento(&temp,'b',blockIdx.x/N,blockIdx.x%N);

		/* ETAPA 3 - SIMULAÇÃO: faz-se a soma de diversas simulações para cada nó que foi expandido */
		/* Faz em passadas, de forma que em um bloco sejam feitas todas as simulações do nó */

		scores[idx] = 0;

		for(i = threadIdx.x; i < num_simulacoes; i+= stride){
			k = 0;
			simulacaoMCTS(temp,&k);
			scores[idx] += k;
		}
		
	}else{
		/* Se não for possível fazer o movimento e as simulações, marca-se essa posição com -666, fazendo com que a média desse nó não seja escolhida. */
		scores[idx]  = -666;
	}
}

int main(){
	struct Estado s;
	int *scores;
	
	int l,m,k;
	int c1,c2;
	int soma;
	int jogadas;
	float maior;
	float media;

	clock_t tempo;
	size_t threadsPerBlock = 256;
	size_t numberOfBlocks = N*N;
	cudaError_t err;

	int tam_scores = numberOfBlocks * threadsPerBlock;

	printf("GameTree - Jogo GO (Adaptado) - Percurso com MCTS (Monte Carlo Tree Search)\n");
	printf("Tabuleiro %d x %d.\n",N,N);
	printf("Considerando %d jogadas.\n",qtd_jogadas);
	printf("Serao feitas %d simulacoes para cada no expandido.\n\n",num_simulacoes);
	
	/* Trabalhando com a Memória Unificada - cada thread terá uma posição do vetor de scores */
	err = cudaMallocManaged(&scores, tam_scores * sizeof(int));
	if(err != cudaSuccess) printf("Error: %s\n",cudaGetErrorString(err));

	printf("INICIANDO O JOGO:\n");
	
	/* Inicializando e imprimindo o primeiro estado de jogo - raiz da árvore */
	inicializaEstadoPadraoHost(&s);
	jogadas = 0;
	
	for(l = 0; l < N; l++){
		for (m = 0; m < N; m++) printf("%c ", s.tabuleiro[l][m]);
		printf("\n");
	}
	
	while(jogadas + 1 <= qtd_jogadas){

		/* Inicializando/resetando o vetor de scores */
		for(l = 0; l < tam_scores; l++) scores[l] = -999;
		
		/* Coleta a jogada, faz o movimento e imprime o tabuleiro */	
		printf("Jogada (p) - linha e coluna: ");
		scanf("%d%d",&c1,&c2);
		
		fazMovimentoHost(&s,'p',c1,c2);
		s.nivel++;
		
		for(l = 0; l < N; l++){
			for (m = 0; m < N; m++) printf("%c ", s.tabuleiro[l][m]);
			printf("\n");
		}
		
		jogadas++;
		
		/* Fazendo o MCTS com a raiz da árvore */
		if(jogadas + 1 <= qtd_jogadas){
			printf("Fazendo MCTS...\n");

			tempo = clock();

			/* Definindo o numero de blocos/threads e fazendo a chamada do kernel */
			mcts<<<numberOfBlocks,threadsPerBlock>>>(s,scores);
			cudaDeviceSynchronize();

			/* Verificação de erros na chamada do kernel */
			err = cudaGetLastError();
			if(err != cudaSuccess) printf("Error: %s\n",cudaGetErrorString(err));

			/* ETAPAS FINAIS DO MCTS - PROPAGAÇÃO: o valor do score das simulações é utilizado para definir o melhor nó */
			/* Obs: dada a forma como o algoritmo foi implementado, não é necessária uma propagação recursiva, com percurso na árvore */
			/* Ao fim, descobre o melhor filho do nó raiz (cujo índice será armazenado em k) */
			maior = -999.0;
			k = 0;

			for(l = 0; l < tam_scores; l+=256){
				soma = 0;

				/* != -999 - quer dizer que a thread fez a simulação */
				for(m = l; m < l+256; m++) if(scores[m] != -999) soma+=scores[m];

				media = (float)soma / num_simulacoes; 

				printf("Media do bloco %d: %f\n",l/256,media);
				if(media > maior){
					maior = media;
					k = l/256;
				}
			} 

			tempo = clock() - tempo;


			/* Imprimindo o resultado do MCTS */
			printf("Resultado do MCTS: indice %d. Tempo: %lf s.\n",k,((double)tempo)/((CLOCKS_PER_SEC)));
			printf("Jogada (b):\n");

			/* Efetivando o melhor movimento encontado na busca e imprimindo o tabuleiro */
			fazMovimentoHost(&s,'b',k/N,k%N);
			s.peca = 'b';
			s.linha = k/N;
			s.coluna = k%N;
			s.nivel++;

			for(l = 0; l < N; l++){
				for (m = 0; m < N; m++) printf("%c ", s.tabuleiro[l][m]);
				printf("\n");
			}

			jogadas++;
		}	

	}
	
	printf("Jogo finalizado - score da maquina (b): %d\n",s.score);

	cudaFree(scores);
	return 0;
}
