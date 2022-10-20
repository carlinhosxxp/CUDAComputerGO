/* 
 Percurso na GameTree - Jogo GO (Adaptado) - utilizando algoritmo MONTE CARLO TREE SEARCH (MCTS)
 Carlos Henrique Rorato Souza - Computação Paralela (2022-1)
 Variação do jogo: 
     - Tabuleiro de tamanho N*N intersecções
     - Captura de somente uma peça por vez, cercada na horizontal e vertical
     - Cálculo do score baseado na quantidade de peças pretas e brancas restantes
 A árvore é representada como um vetor de structs, cada um representando um nó.
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

#define N 19
#define qtd_jogadas N*N
#define num_simulacoes 100

/*
 Estrutura Estado, que representa um nó da árvore, com:
	- Tabuleiro
	- Pai (guarda o índice do pai)
	- Filhos (guarda os índices dos filhos - o tamanho do vetor vem do pior caso)
	- Score (pontuação, calculada em função específica)
	- Peca (qual peça foi colocada nesta jogada)
	- Linha, Coluna (em qual posição essa peça foi colocada)
	- Nível (o nível da árvore no qual o nó está)
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
 Função que inicializa um estado com os valores padrão, 
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
 Função que calcula o score, percorrendo o tabuleiro
 do estado. Calcula-se o score do jogador que usa as
 pedras brancas. O cálculo é feito a partir da diferença
 entre as peças brancas e pretas restantes no tabuleiro.
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
 Função que insere uma peça no tabuleiro e verifica se essa inserção
 captura alguma peça da cor oposta, verificando as quatro direções
 (cima, baixo, direita e esqueda). Caso alguma peça esteja nas bordas
 do tabuleiro, ela já é considerada cercada na direção da(s) borda(s).
 A função verifica também se o movimento é suicida, isto é, se a peça 
 foi inserida numa posição onde é capturada.
*/
void fazMovimento(struct Estado *s, char peca, int i, int j){ //peca pode ser "b" ou "p"
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


/*
 Função auxiliar que copia o tabuleiro de um nó para outro
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
 Função auxiliar para o MCTS que simula o caminho até o final da árvore (fim de jogo),
 sorteando caminhos na árovore a partir do estado s (jogadas aleatórias)
 e retornando ao final o score desse jogo.
*/
int simulacaoMCTS(struct Estado s){
	char peca = 'p';
	int i,j;
	
	for(i = 0; i < N*N - s.nivel; i++){		
		/* Sorteia-se uma posição válida no tabuleiro e se faz a jogada */
		do{
			j = rand()%(N*N);
		}while(s.tabuleiro[j/N][j%N] != '-');
		fazMovimento(&s, peca,j/N,j%N);
		
		/* Inverte-se a peça a ser jogada no próximo nível */
		peca = peca == 'b'? 'p': 'b';
	}
	calculaScore(&s);
	return s.score;
}

/*
 Função MCTS (Monte Carlo Tree Search), que recebe a árvore (vetor) e
 retorna o índice do vetor que possui a melhor jogada a ser feita.
 Este algoritmo constrói e armazena somente uma árvore parcial, com um nó raiz e seus filhos.
 Está baseado na seguinte sequência: seleção da raiz e expansão dos filhos do nó raiz,
 simulação de caminhos de jogo aleatórios para cada nó que foi gerado na expansão,
 propagação do score desta configuração de jogo e a seleção da jogada (caminho
 na árvore) que trará o score mais favorável a partir das simulações.
*/
int mcts(struct Estado *gameTree){
	
	/* ETAPA 1 - SELEÇÃO: seleciona o nó raiz */
	int controle = 0;
	int j = 0;
	int indice = 0;
	int indice_arvore = 1;
	int qtd_filhos = N*N - gameTree[indice].nivel;
	int maior_score = -999;
	int indice_maior_score = -1;
	int score_atual = 0;
	int i = 0;
	
	/* ETAPA 2 - EXPANSÃO: faz a expansão de todos os filhos do nó selecionado (raiz) - esta é a política de expansão*/
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
			
			/* ETAPA 3 - SIMULAÇÃO: faz-se a média de diversas simulações para cada nó que foi expandido */
			for(i = 0; i < num_simulacoes; i++){
				score_atual += simulacaoMCTS(gameTree[indice_arvore]);
			}
			score_atual = score_atual / num_simulacoes;
			
			/* ETAPA 4 - PROPAGAÇÃO: o valor do score das simulações é utilizado para definir o melhor nó */
			/* Obs: dada a forma como o algoritmo foi implementado, não é necessária uma propagação recursiva, com percurso na árvore */
			if(score_atual > maior_score){
				maior_score = score_atual;
				indice_maior_score = indice_arvore;
			}
			
			controle++;
			indice_arvore++;
		}
		j++;
	}
	
	/* ETAPA 5 - Retorna o melhor filho do nó raiz */
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
	
	/* Inicializando o primeiro estado de jogo - raiz da árvore */
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
