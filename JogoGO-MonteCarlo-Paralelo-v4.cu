/* 
 Percurso na GameTree - Jogo GO (Adaptado) - utilizando algoritmo MONTE CARLO TREE SEARCH (MCTS)
 Vers�o Paralela (CUDA)
 Carlos Henrique Rorato Souza - Computa��o Paralela (2022-1)
 Varia��o do jogo: 
     - Tabuleiro de tamanho N*N intersec��es
     - Captura de somente uma pe�a por vez, cercada na horizontal e vertical
     - C�lculo do score baseado na quantidade de pe�as pretas e brancas restantes
 � armazenado somente o estado atual de jogo, e as simula��es s�o feitas a partir deste estado.
 Paraleliza��o: cada bloco far� o processamento (simula��es) referente � um n� filho da raiz.
*/

/* 
 Defini��es iniciais:
	- N � o tamanho do tabuleiro
	- qtd_jogoadas define a quantidade de jogadas (n�veis da �rvore)
	- num_simulacoes define a quantidade de simula��es que o MCTS far� para cada n�
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
 Estrutura Estado, que representa um n� da �rvore, com:
	- Tabuleiro
	- Peca (qual pe�a foi colocada nesta jogada)
	- Score (pontua��o, calculada em fun��o espec�fica)
	- Linha, Coluna (em qual posi��o essa pe�a foi colocada)
	- N�vel (o n�vel da �rvore no qual o n� est�)
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
 Fun��o que faz a gera��o de n�meros pseudoaleat�rios em CUDA,
 para uso na etapa de simula��o.
 Para a inicializa��o, informamos a semente para gera��o dos n�meros.
*/
__device__ void random(int* resultado, int limite) {
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	curandState_t state;
	curand_init(clock(), id, 0, &state);
	*resultado = curand(&state) % limite;
}

/*
 Fun��o que inicializa um estado com os valores padr�o, 
 sem pai e sem filhos, com tabuleiro vazio.
 Como ela � executada tanto na GPU quanto na CPU, est�o dispon�veis as duas vers�es.
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
 Fun��o que calcula o score, percorrendo o tabuleiro
 do estado. Calcula-se o score do jogador que usa as
 pedras brancas. O c�lculo � feito a partir da diferen�a
 entre as pe�as brancas e pretas restantes no tabuleiro.
 Como ela � executada tanto na GPU quanto na CPU, est�o dispon�veis as duas vers�es.
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
 Fun��o que insere uma pe�a no tabuleiro e verifica se essa inser��o
 captura alguma pe�a da cor oposta, verificando as quatro dire��es
 (cima, baixo, direita e esqueda). Caso alguma pe�a esteja nas bordas
 do tabuleiro, ela j� � considerada cercada na dire��o da(s) borda(s).
 A fun��o verifica tamb�m se o movimento � suicida, isto �, se a pe�a 
 foi inserida numa posi��o onde � capturada.
 Como ela � executada tanto na GPU quanto na CPU, est�o dispon�veis as duas vers�es.
*/
__device__ void fazMovimento(struct Estado *s, char peca, int i, int j){ //peca pode ser "b" ou "p"
	int contador;
	char oponente = peca == 'b'? 'p': 'b';
	
	if(s->tabuleiro[i][j] == '-'){
		s->tabuleiro[i][j] = peca;
	
		/* olhado para cima (j-1) e verificando se a inser��o da pe�a cercou a pe�a de cima */
		contador = 0;
		if(j-1>=0 && s->tabuleiro[i][j-1] == oponente){
			if (j-1 + 1 < N) {if(s->tabuleiro[i][j-1 + 1] == peca) contador++;} else contador++;
			if (j-1 - 1 >=0) {if(s->tabuleiro[i][j-1 - 1] == peca) contador++;} else contador++;
			if (i+1 < N) {if(s->tabuleiro[i + 1][j-1] == peca) contador++;} else contador++;
			if (i-1 >=0) {if(s->tabuleiro[i - 1][j-1] == peca) contador++;} else contador++;
			
			/* Se a pe�a est� cercada, � capturada */
			if(contador == 4) s->tabuleiro[i][j-1] = '-';
		}
		
		/* olhado para baixo (j+1) e verificando se a inser��o da pe�a cercou a pe�a de baixo */
		contador = 0;
		if(j+1<N && s->tabuleiro[i][j+1] == oponente){
			if (j+1 + 1 < N) {if(s->tabuleiro[i][j+1 + 1] == peca) contador++;} else contador++;
			if (j+1 - 1 >=0) {if(s->tabuleiro[i][j+1 - 1] == peca) contador++;} else contador++;
			if (i+1 < N) {if(s->tabuleiro[i + 1][j+1] == peca) contador++;} else contador++;
			if (i-1 >=0) {if(s->tabuleiro[i - 1][j+1] == peca) contador++;} else contador++;
			
			/* Se a pe�a est� cercada, � capturada */
			if(contador == 4) s->tabuleiro[i][j+1] = '-';
		}
		
		/* olhado para a esquerda (i-1) e verificando se a inser��o da pe�a cercou a pe�a da esquerda */
		contador = 0;
		if(i-1>=0 && s->tabuleiro[i-1][j] == oponente){
			if (j + 1 < N) {if(s->tabuleiro[i - 1][j + 1] == peca) contador++;} else contador++;
			if (j - 1 >=0) {if(s->tabuleiro[i - 1][j - 1] == peca) contador++;} else contador++;
			if (i - 1 + 1 < N) {if(s->tabuleiro[i -1 + 1][j] == peca) contador++;} else contador++;
			if (i - 1 - 1 >=0) {if(s->tabuleiro[i -1 - 1][j] == peca) contador++;} else contador++;
			
			/* Se a pe�a est� cercada, � capturada */
			if(contador == 4) s->tabuleiro[i-1][j] = '-';
		}
		
		/* olhado para a direita (i+1) e verificando se a inser��o da pe�a cercou a pe�a da direita */
		contador = 0;
		if(i+1<N && s->tabuleiro[i+1][j] == oponente){
			if (j + 1 < N) {if(s->tabuleiro[i + 1][j + 1] == peca) contador++;} else contador++;
			if (j - 1 >=0) {if(s->tabuleiro[i + 1][j - 1] == peca) contador++;} else contador++;
			if (i + 1 + 1 < N) {if(s->tabuleiro[i + 1 + 1][j] == peca) contador++;} else contador++;
			if (i + 1 - 1 >=0) {if(s->tabuleiro[i + 1 - 1][j] == peca) contador++;} else contador++;
			
			/* Se a pe�a est� cercada, � capturada */
			if(contador == 4) s->tabuleiro[i+1][j] = '-';
		}
		
		/* Avaliando se a jogada � suicida, ou seja, se a pe�a foi inserida numa posi��o onde � capturada */
		contador = 0;
		if(j + 1 < N){  if(s->tabuleiro[i][j+1] == oponente) contador++; } else contador++;
		if(j - 1 >= 0){ if(s->tabuleiro[i][j-1] == oponente) contador++; } else contador++;
		if(i + 1 < N){  if(s->tabuleiro[i+1][j] == oponente) contador++; } else contador++;
		if(i - 1 >= 0){ if(s->tabuleiro[i-1][j] == oponente) contador++; } else contador++;
		
		/* Se a pe�a est� cercada, � capturada */
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
	
		/* olhado para cima (j-1) e verificando se a inser��o da pe�a cercou a pe�a de cima */
		contador = 0;
		if(j-1>=0 && s->tabuleiro[i][j-1] == oponente){
			if (j-1 + 1 < N) {if(s->tabuleiro[i][j-1 + 1] == peca) contador++;} else contador++;
			if (j-1 - 1 >=0) {if(s->tabuleiro[i][j-1 - 1] == peca) contador++;} else contador++;
			if (i+1 < N) {if(s->tabuleiro[i + 1][j-1] == peca) contador++;} else contador++;
			if (i-1 >=0) {if(s->tabuleiro[i - 1][j-1] == peca) contador++;} else contador++;
			
			/* Se a pe�a est� cercada, � capturada */
			if(contador == 4) s->tabuleiro[i][j-1] = '-';
		}
		
		/* olhado para baixo (j+1) e verificando se a inser��o da pe�a cercou a pe�a de baixo */
		contador = 0;
		if(j+1<N && s->tabuleiro[i][j+1] == oponente){
			if (j+1 + 1 < N) {if(s->tabuleiro[i][j+1 + 1] == peca) contador++;} else contador++;
			if (j+1 - 1 >=0) {if(s->tabuleiro[i][j+1 - 1] == peca) contador++;} else contador++;
			if (i+1 < N) {if(s->tabuleiro[i + 1][j+1] == peca) contador++;} else contador++;
			if (i-1 >=0) {if(s->tabuleiro[i - 1][j+1] == peca) contador++;} else contador++;
			
			/* Se a pe�a est� cercada, � capturada */
			if(contador == 4) s->tabuleiro[i][j+1] = '-';
		}
		
		/* olhado para a esquerda (i-1) e verificando se a inser��o da pe�a cercou a pe�a da esquerda */
		contador = 0;
		if(i-1>=0 && s->tabuleiro[i-1][j] == oponente){
			if (j + 1 < N) {if(s->tabuleiro[i - 1][j + 1] == peca) contador++;} else contador++;
			if (j - 1 >=0) {if(s->tabuleiro[i - 1][j - 1] == peca) contador++;} else contador++;
			if (i - 1 + 1 < N) {if(s->tabuleiro[i -1 + 1][j] == peca) contador++;} else contador++;
			if (i - 1 - 1 >=0) {if(s->tabuleiro[i -1 - 1][j] == peca) contador++;} else contador++;
			
			/* Se a pe�a est� cercada, � capturada */
			if(contador == 4) s->tabuleiro[i-1][j] = '-';
		}
		
		/* olhado para a direita (i+1) e verificando se a inser��o da pe�a cercou a pe�a da direita */
		contador = 0;
		if(i+1<N && s->tabuleiro[i+1][j] == oponente){
			if (j + 1 < N) {if(s->tabuleiro[i + 1][j + 1] == peca) contador++;} else contador++;
			if (j - 1 >=0) {if(s->tabuleiro[i + 1][j - 1] == peca) contador++;} else contador++;
			if (i + 1 + 1 < N) {if(s->tabuleiro[i + 1 + 1][j] == peca) contador++;} else contador++;
			if (i + 1 - 1 >=0) {if(s->tabuleiro[i + 1 - 1][j] == peca) contador++;} else contador++;
			
			/* Se a pe�a est� cercada, � capturada */
			if(contador == 4) s->tabuleiro[i+1][j] = '-';
		}
		
		/* Avaliando se a jogada � suicida, ou seja, se a pe�a foi inserida numa posi��o onde � capturada */
		contador = 0;
		if(j + 1 < N){  if(s->tabuleiro[i][j+1] == oponente) contador++; } else contador++;
		if(j - 1 >= 0){ if(s->tabuleiro[i][j-1] == oponente) contador++; } else contador++;
		if(i + 1 < N){  if(s->tabuleiro[i+1][j] == oponente) contador++; } else contador++;
		if(i - 1 >= 0){ if(s->tabuleiro[i-1][j] == oponente) contador++; } else contador++;
		
		/* Se a pe�a est� cercada, � capturada */
		if(contador == 4) s->tabuleiro[i][j] = '-';
		
		/* Ao final, calcula o score */
		calculaScoreHost(s);
	}
}

/*
 Fun��o auxiliar que copia o tabuleiro de um n� para outro.
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
 Fun��o auxiliar para o MCTS que simula o caminho at� o final da �rvore (fim de jogo),
 sorteando caminhos na �rovore a partir do estado s (jogadas aleat�rias)
 e retornando ao final o score desse jogo.
*/
__device__ void simulacaoMCTS(struct Estado s, int *resultado){
	char peca = 'p';
	int i,j;
	
	for(i = 0; i < N*N - s.nivel; i++){		
		/* Sorteia-se uma posi��o v�lida no tabuleiro e se faz a jogada */
		do{
			random(&j,N*N);
		}while(s.tabuleiro[j/N][j%N] != '-');
		fazMovimento(&s, peca,j/N,j%N);
		
		/* Inverte-se a pe�a a ser jogada no pr�ximo n�vel */
		peca = peca == 'b'? 'p': 'b';
	}
	calculaScore(&s);
	*resultado = s.score;
}

/*
 Fun��o MCTS (Monte Carlo Tree Search), que recebe o estado atual da �rvore e
 preenche o vetor de scores, com o somat�rio de scores da simula��o (para cada n�).
 Este algoritmo constr�i somente uma �rvore parcial, com um n� raiz e seus filhos.
 Est� baseado na seguinte sequ�ncia: sele��o da raiz e expans�o dos filhos do n� raiz,
 simula��o de caminhos de jogo aleat�rios para cada n� que foi gerado na expans�o,
 propaga��o do score desta configura��o de jogo e a sele��o da jogada (caminho
 na �rvore) que trar� o score mais favor�vel a partir das simula��es. Estas duas �ltimas etapas
 n�o s�o feitas nessa fun��o, mas dentro da pr�pria fun��o principal, a partir do vetor de scores.
 Ela � executada na GPU. Nesse contexto, a solu��o foi estruturada de maneira que cada bloco
 processe as opera��es referentes a um n� filho.
 Os filhos n�o s�o armazenados, mas a simula��o indicar� a melhor jogada, que ser� concretizada
 dentro do tabuleiro do jogo (estado atual), na fun��o principal.
*/
__global__ void mcts(struct Estado gameTree, int *scores){
	
	/* ETAPA 1 - SELE��O: seleciona o n� raiz e declara/inicializa vari�veis importantes para a fun��o */
	int i = 0;
	int k = 0;
	int stride = blockDim.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	struct Estado temp;

	/* ETAPA 2 - EXPANS�O: faz a expans�o de todos os filhos do n� selecionado (raiz) - esta � a pol�tica de expans�o */
	/* Cada bloco ser� um filho do n� raiz, armazenado de forma tempor�ria */
	inicializaEstadoPadrao(&temp);
	
	/* O filho continua o jogo do pai */
	copiarTabuleiro(gameTree,&temp);
	
	if(temp.tabuleiro[blockIdx.x/N][blockIdx.x%N] == '-'){
		fazMovimento(&temp,'b',blockIdx.x/N,blockIdx.x%N);

		/* ETAPA 3 - SIMULA��O: faz-se a soma de diversas simula��es para cada n� que foi expandido */
		/* Faz em passadas, de forma que em um bloco sejam feitas todas as simula��es do n� */

		scores[idx] = 0;

		for(i = threadIdx.x; i < num_simulacoes; i+= stride){
			k = 0;
			simulacaoMCTS(temp,&k);
			scores[idx] += k;
		}
		
	}else{
		/* Se n�o for poss�vel fazer o movimento e as simula��es, marca-se essa posi��o com -666, fazendo com que a m�dia desse n� n�o seja escolhida. */
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
	
	/* Trabalhando com a Mem�ria Unificada - cada thread ter� uma posi��o do vetor de scores */
	err = cudaMallocManaged(&scores, tam_scores * sizeof(int));
	if(err != cudaSuccess) printf("Error: %s\n",cudaGetErrorString(err));

	printf("INICIANDO O JOGO:\n");
	
	/* Inicializando e imprimindo o primeiro estado de jogo - raiz da �rvore */
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
		
		/* Fazendo o MCTS com a raiz da �rvore */
		if(jogadas + 1 <= qtd_jogadas){
			printf("Fazendo MCTS...\n");

			tempo = clock();

			/* Definindo o numero de blocos/threads e fazendo a chamada do kernel */
			mcts<<<numberOfBlocks,threadsPerBlock>>>(s,scores);
			cudaDeviceSynchronize();

			/* Verifica��o de erros na chamada do kernel */
			err = cudaGetLastError();
			if(err != cudaSuccess) printf("Error: %s\n",cudaGetErrorString(err));

			/* ETAPAS FINAIS DO MCTS - PROPAGA��O: o valor do score das simula��es � utilizado para definir o melhor n� */
			/* Obs: dada a forma como o algoritmo foi implementado, n�o � necess�ria uma propaga��o recursiva, com percurso na �rvore */
			/* Ao fim, descobre o melhor filho do n� raiz (cujo �ndice ser� armazenado em k) */
			maior = -999.0;
			k = 0;

			for(l = 0; l < tam_scores; l+=256){
				soma = 0;

				/* != -999 - quer dizer que a thread fez a simula��o */
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
