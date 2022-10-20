/* 
 Percurso na GameTree - Jogo GO (Adaptado) - utilizando algoritmo MONTE CARLO TREE SEARCH (MCTS)
 Carlos Henrique Rorato Souza - Computa��o Paralela (2022-1)
 Varia��o do jogo: 
     - Tabuleiro de tamanho N*N intersec��es
     - Captura de somente uma pe�a por vez, cercada na horizontal e vertical
     - C�lculo do score baseado na quantidade de pe�as pretas e brancas restantes
 A �rvore � representada como um vetor de structs, cada um representando um n�.
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

#define N 19
#define qtd_jogadas N*N
#define num_simulacoes 100

/*
 Estrutura Estado, que representa um n� da �rvore, com:
	- Tabuleiro
	- Pai (guarda o �ndice do pai)
	- Filhos (guarda os �ndices dos filhos - o tamanho do vetor vem do pior caso)
	- Score (pontua��o, calculada em fun��o espec�fica)
	- Peca (qual pe�a foi colocada nesta jogada)
	- Linha, Coluna (em qual posi��o essa pe�a foi colocada)
	- N�vel (o n�vel da �rvore no qual o n� est�)
*/
struct Estado{
	char tabuleiro[N][N];
	int pai;
	int filhos[N*N];
	int score;
	char peca;
	int linha,coluna;
	int nivel;
};

/*
 Fun��o que inicializa um estado com os valores padr�o, 
 sem pai e sem filhos, com tabuleiro vazio.
*/
void inicializaEstadoPadrao(struct Estado *s){
	int i,j;
	for(i=0; i<N; i++) for(j=0; j<N; j++) s->tabuleiro[i][j] = '-';
	s->pai = -1;
	for(i=0; i<=N*N; i++) s->filhos[i] = -1;
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
*/
void calculaScore(struct Estado *s){
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
*/
void fazMovimento(struct Estado *s, char peca, int i, int j){ //peca pode ser "b" ou "p"
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


/*
 Fun��o auxiliar que copia o tabuleiro de um n� para outro
*/
void copiarTabuleiro(struct Estado *original, struct Estado *copia){
	int i,j;
	for(i=0; i<N; i++){
		for(j=0; j<N; j++){
			copia->tabuleiro[i][j] = original->tabuleiro[i][j];
		}
	}
}

/*
 Fun��o auxiliar para o MCTS que simula o caminho at� o final da �rvore (fim de jogo),
 sorteando caminhos na �rovore a partir do estado s (jogadas aleat�rias)
 e retornando ao final o score desse jogo.
*/
int simulacaoMCTS(struct Estado s){
	char peca = 'p';
	int i,j;
	
	for(i = 0; i < N*N - s.nivel; i++){		
		/* Sorteia-se uma posi��o v�lida no tabuleiro e se faz a jogada */
		do{
			j = rand()%(N*N);
		}while(s.tabuleiro[j/N][j%N] != '-');
		fazMovimento(&s, peca,j/N,j%N);
		
		/* Inverte-se a pe�a a ser jogada no pr�ximo n�vel */
		peca = peca == 'b'? 'p': 'b';
	}
	calculaScore(&s);
	return s.score;
}

/*
 Fun��o MCTS (Monte Carlo Tree Search), que recebe a �rvore (vetor) e
 retorna o �ndice do vetor que possui a melhor jogada a ser feita.
 Este algoritmo constr�i e armazena somente uma �rvore parcial, com um n� raiz e seus filhos.
 Est� baseado na seguinte sequ�ncia: sele��o da raiz e expans�o dos filhos do n� raiz,
 simula��o de caminhos de jogo aleat�rios para cada n� que foi gerado na expans�o,
 propaga��o do score desta configura��o de jogo e a sele��o da jogada (caminho
 na �rvore) que trar� o score mais favor�vel a partir das simula��es.
*/
int mcts(struct Estado *gameTree){
	
	/* ETAPA 1 - SELE��O: seleciona o n� raiz */
	int controle = 0;
	int j = 0;
	int indice = 0;
	int indice_arvore = 1;
	int qtd_filhos = N*N - gameTree[indice].nivel;
	int maior_score = -999;
	int indice_maior_score = -1;
	int score_atual = 0;
	int i = 0;
	
	/* ETAPA 2 - EXPANS�O: faz a expans�o de todos os filhos do n� selecionado (raiz) - esta � a pol�tica de expans�o*/
	while(controle < qtd_filhos && j < N*N){
		inicializaEstadoPadrao(&gameTree[indice_arvore]);
		
		/* O filho continua o jogo do pai */
		copiarTabuleiro(&gameTree[indice],&gameTree[indice_arvore]);
		
		if(gameTree[indice_arvore].tabuleiro[j/N][j%N] == '-'){;
			fazMovimento(&gameTree[indice_arvore],'b',j/N,j%N);
			gameTree[indice_arvore].peca = 'b';
			gameTree[indice_arvore].linha = j/N;
			gameTree[indice_arvore].coluna = j%N;
			gameTree[indice_arvore].pai = indice;
			gameTree[indice_arvore].nivel = gameTree[indice].nivel + 1;
			gameTree[indice].filhos[controle] = indice_arvore;
			
			/* ETAPA 3 - SIMULA��O: faz-se a m�dia de diversas simula��es para cada n� que foi expandido */
			for(i = 0; i < num_simulacoes; i++){
				score_atual += simulacaoMCTS(gameTree[indice_arvore]);
			}
			score_atual = score_atual / num_simulacoes;
			
			/* ETAPA 4 - PROPAGA��O: o valor do score das simula��es � utilizado para definir o melhor n� */
			/* Obs: dada a forma como o algoritmo foi implementado, n�o � necess�ria uma propaga��o recursiva, com percurso na �rvore */
			if(score_atual > maior_score){
				maior_score = score_atual;
				indice_maior_score = indice_arvore;
			}
			
			controle++;
			indice_arvore++;
		}
		j++;
	}
	
	/* ETAPA 5 - Retorna o melhor filho do n� raiz */
	return indice_maior_score;
}

int main(){
	struct Estado *s;
	int l,m,k;
	int c1,c2;
	int jogadas = 0;
	clock_t tempo;
	
	printf("GameTree - Jogo GO (Adaptado) - Percurso com MCTS (Monte Carlo Tree Search)\n");
	printf("Tabuleiro %d x %d.\n",N,N);
	printf("Considerando %d jogadas.\n",qtd_jogadas);
	printf("Serao feitas %d simulacoes para cada no expandido.\n\n",num_simulacoes);
	
	s = malloc(N * N * sizeof(struct Estado));
	
	printf("INICIANDO O JOGO:\n");
	
	/* Inicializando o primeiro estado de jogo - raiz da �rvore */
	k=0;
	jogadas = 0;
	
	inicializaEstadoPadrao(&s[k]);
	for(l = 0; l < N; l++){
			for (m = 0; m < N; m++) printf("%c ", s[k].tabuleiro[l][m]);
			printf("\n");
	}
	
	while(jogadas + 1 <= qtd_jogadas){
			
		printf("Jogada (p) - linha e coluna: ");
		scanf("%d%d",&c1,&c2);
		
		fazMovimento(&s[k],'p',c1,c2);
		s[k].nivel++;
		
		for(l = 0; l < N; l++){
			for (m = 0; m < N; m++) printf("%c ", s[k].tabuleiro[l][m]);
			printf("\n");
		}
		
		jogadas++;
		
		if(jogadas + 1 <= qtd_jogadas){
			printf("Fazendo MCTS...\n");
			tempo = clock();
			k = mcts(s);
			tempo = clock() - tempo;
			printf("Resultado do MCTS: indice %d. Tempo: %lf s.\n",k,((double)tempo)/((CLOCKS_PER_SEC)));
			printf("Jogada (b):\n");
			for(l = 0; l < N; l++){
				for (m = 0; m < N; m++) printf("%c ", s[k].tabuleiro[l][m]);
				printf("\n");
			}
			jogadas++;
			s[0] = s[k];
			k = 0;
		}
	}
	
	printf("Jogo finalizado - score da maquina (b): %d\n",s[k].score);
	
	free(s);
	return 0;
}
