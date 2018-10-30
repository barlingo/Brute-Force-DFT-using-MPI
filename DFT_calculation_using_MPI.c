/*
	This program generates a random matrix given its argument inputs and calculates its DFT real and imaginary part using mpi.
*/
#include "mpi.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define DEFAULT_N_ROWS 100
#define DEFAULT_N_COLUMNS 100
#define PI 3.14159265359f

#define SEND_MATRIX_TAG 0
#define RECV_MATRIX_TAG 1
#define PRINT_MATRIX_TAG 2

#define MPI_MASTER_SEND 0
#define MPI_MASTER_RECV 1

#define CALC_DFT_TAG 3
#define INIT_TAG 4
#define DECOMPOSE 0
#define COMPOSE 1

#define PRINT true		// Activates or deactivate printing matrices
#define LOGPID true		// Activates or deactivates printing P_ID messages

/*
 Global variables
*/


int numProcMPI, processMPI, processorNameLenMPI;
char processorNameMPI[MPI_MAX_PROCESSOR_NAME];
int root = 0;
double startwtime;
double endwtime;
MPI_Status status;

/*
 Function prototypes
*/

void randomizeMatrix(double**, int, int);
void printMatrix(double**, int, int);
void printMatrixRealImag(double**, double**, int, int);
void slideMatrix(double**, double**, int, int, int, int, int);
void slideSendMatrix(double**, int, int, int, int, int);

double** allocateMatrix(int, int);
void calcMatrixSubDim(int, int, int*, int*, int);
void calcDFTMatrix(double**, double**, double**,int, int);
void initMPI();
void logMessagePID(int,int,int);
void enteredMatrixDimCheck(int*,int*);


int main(int argc, char *argv[])
{

	initMPI();

	//Pointers to Matrices to be used by root process
	double** Matrix = NULL;
	double** MatrixReal = NULL;
	double** MatrixImag = NULL;

	//Pointers to Matrices to be used by all processes
	double** subMatrix = NULL;
	double** subMatrixReal = NULL;
	double** subMatrixImag = NULL;

	int n_rows;
	int n_columns;
	int sub_n_rows;
	int sub_n_columns;


	if (processMPI == root) {
		startwtime = MPI_Wtime();
		if (argc > 2) {
			n_rows = atoi(argv[1]);
			n_columns = atoi(argv[2]);
			enteredMatrixDimCheck(&n_rows, &n_columns);
		}
		else {
			n_rows = numProcMPI;
			n_columns = numProcMPI;
			printf("Matrix values set to default %d x %d\n", n_rows, n_columns);
		}
		if (numProcMPI > 1) {
			calcMatrixSubDim(n_rows, n_columns, &sub_n_rows, &sub_n_columns, numProcMPI);
		}

	}

	// Needs to broadcast calculated in PID0 sub rows and sub columns to all processes 
	MPI_Bcast(&sub_n_rows, sizeof(int), MPI_INT, root, MPI_COMM_WORLD);
	MPI_Bcast(&sub_n_columns, sizeof(int), MPI_INT, root, MPI_COMM_WORLD);


	subMatrix = allocateMatrix(sub_n_rows, sub_n_columns);

	if (processMPI == root) {
		Matrix = allocateMatrix(n_rows, n_columns);
		// Create and initialize Matrix with random numbers
		randomizeMatrix(Matrix, n_rows, n_columns);
		printMatrix(Matrix, n_rows, n_columns); 
		//Take first Matrix square for root
		slideMatrix(Matrix, subMatrix, sub_n_rows, sub_n_columns, 0, 0, DECOMPOSE);
		//Cut Matrix into sub matrices and sends them to all processors
		slideSendMatrix(Matrix, sub_n_rows, sub_n_columns, n_rows, n_columns, MPI_MASTER_SEND);
	}
	if (processMPI != root) {
		logMessagePID(root, processMPI, RECV_MATRIX_TAG);
		MPI_Recv(&(subMatrix[0][0]), sub_n_rows*sub_n_columns, MPI_DOUBLE, root, MPI_MASTER_SEND, MPI_COMM_WORLD, &status);
	}

	// All processes calculates sub Matrix DFT
	subMatrixReal = allocateMatrix(sub_n_rows, sub_n_columns);
	subMatrixImag = allocateMatrix(sub_n_rows, sub_n_columns);
	calcDFTMatrix(subMatrix, subMatrixReal, subMatrixImag, sub_n_rows, sub_n_columns);
	printMatrixRealImag(subMatrixReal, subMatrixImag, sub_n_rows, sub_n_columns);
	free(subMatrix); // No need for subMatrix after this point , values stored in subMatrixReal and subMatrixImag

	// Gather all real parts of Matrix in root
	if (processMPI == root) {
		MatrixReal = allocateMatrix(n_rows, n_columns);
		//Take first squares
		slideMatrix(MatrixReal, subMatrixReal, sub_n_rows, sub_n_columns, 0, 0, COMPOSE);
		slideSendMatrix(MatrixReal, sub_n_rows, sub_n_columns, n_rows, n_columns, MPI_MASTER_RECV);
	} 
	else if (processMPI != root) {
		logMessagePID(processMPI, root, SEND_MATRIX_TAG);
		MPI_Send(&(subMatrixReal[0][0]), sub_n_rows*sub_n_columns, MPI_DOUBLE, root, MPI_MASTER_RECV, MPI_COMM_WORLD);
	}
	free(subMatrixReal);

	// Gather all imaginary parts of Matrix in root
	if (processMPI == root) {
		MatrixImag = allocateMatrix(n_rows, n_columns);
		//Take first squares
		slideMatrix(MatrixImag, subMatrixImag, sub_n_rows, sub_n_columns, 0, 0, COMPOSE);
		slideSendMatrix(MatrixImag, sub_n_rows, sub_n_columns, n_rows, n_columns, MPI_MASTER_RECV);
	}
	else if (processMPI != root) {
		logMessagePID(processMPI, root, SEND_MATRIX_TAG);
		MPI_Send(&(subMatrixImag[0][0]), sub_n_rows*sub_n_columns, MPI_DOUBLE, root, MPI_MASTER_RECV, MPI_COMM_WORLD);
	}
	free(subMatrixImag);

	if (processMPI == root) {

		printMatrixRealImag(MatrixReal, MatrixImag, n_rows, n_columns);
		//printf("\tP\tR\tC\tRxC\tKB\tTs\n");
		printf("\t%d\t%d\t%d\t%d\t%.3f\t%f\n", numProcMPI, n_rows, n_columns, n_rows* n_columns,  (((float)n_rows* (float)n_columns * sizeof(double)) / 1024 ), MPI_Wtime() - startwtime);

		// Sequential calculation
		//startwtime = MPI_Wtime();
		//calcDFTMatrix(Matrix,MatrixReal,MatrixImag,n_rows,n_columns); //Sequencial calculation
		//printf("\t%d\t%d\t%d\t%d\t%.3f\t%f\n", 1, n_rows, n_columns, n_rows* n_columns, (((float)n_rows* (float)n_columns * sizeof(double)) / 1024), MPI_Wtime() - startwtime);
		free(Matrix);
		free(MatrixReal);
		free(MatrixImag);
	}

	MPI_Finalize();



	return 0;
}

/*
 * Function:  initMPI
 * --------------------
 *	MPI initialization.
 *	returns:	null
 */
void initMPI()
{
	int rc = MPI_Init(NULL, NULL);
	if (rc != MPI_SUCCESS) {
		printf("Error stating MPI program. Terminating.\n");
		MPI_Abort(MPI_COMM_WORLD, rc);
	}

	MPI_Comm_size(MPI_COMM_WORLD, &numProcMPI);
	MPI_Comm_rank(MPI_COMM_WORLD, &processMPI);
	MPI_Get_processor_name(processorNameMPI, &processorNameLenMPI);
	if (processMPI == 0) {
	//	printf("Running program in %d processes in %s.\n", numProcMPI, processorNameMPI);
	}
	logMessagePID(processMPI, 0, INIT_TAG);

}

/*
 * Function:  Calculate DFT of a MAtrix and saves it into DFTMatrixReal part and DFTMatrixIm part
 * --------------------
 *	Average a Matrix.
 *	Matrix[][]:	Matrix to calculate .
 *	start:		start of Matrix.
 *	end:		end of Matrix
 *	returns:	null
 */
void calcDFTMatrix(double** Matrix_in, double** DFTMatrixReal, double** DFTMatrixIm, int rows, int columns)
{
	logMessagePID(processMPI, 0, CALC_DFT_TAG);
	int i, j, k, l;
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < columns; j++)
		{
			double RealPart = 0;
			double ImagPart = 0;
			for (k = 0; k < rows; k++)
			{
				for (l = 0; l < columns; l++)
				{

					double x = -2.0f*PI*i*k / rows;
					double y = -2.0f*PI*j*l / columns;
					RealPart += Matrix_in[k][l] * cos(x + y);
					ImagPart += Matrix_in[k][l] * sin(x + y);
				}
			}
			DFTMatrixReal[i][j] = RealPart;
			DFTMatrixIm[i][j] = ImagPart;
		}
	}
}


/*
 * Function:  slideSendMatrix
 * --------------------
 *	Given a Matrix it cuts into equal sizes and it places them into bufferMatrix
 *	Matrix:			Main Matrix to be cutted and sent.
 *	sub_rows:		Sub matrix number of rows.
 *	sub_columns:	Sub matrix number of columns.
 *	rows:			Main matrix rows.
 *	columns:		Main matrix columns.
 *	returns:		null
 */
void slideSendMatrix(double **Matrix, int sub_rows, int sub_columns, int rows, int columns, int Action)
{
	double** bufferMatrix = allocateMatrix(sub_rows, sub_columns);

	int slave = 0, i ,j ; 
	for (i = 0; i < ( rows / sub_rows ) ; i++) {
		for (j = 0; j < (columns / sub_columns); j++) {
			if (slave != root) {	// discard first slave since it the square corresponding to root
				if (Action == MPI_MASTER_SEND) {
					slideMatrix(Matrix, bufferMatrix, sub_rows, sub_columns, i * sub_rows, j * sub_columns, DECOMPOSE);
					logMessagePID(root, slave, SEND_MATRIX_TAG);
					printMatrix(bufferMatrix, sub_rows, sub_columns);
					MPI_Send(&(bufferMatrix[0][0]), sub_rows*sub_columns, MPI_DOUBLE, slave, MPI_MASTER_SEND, MPI_COMM_WORLD);
				}
				else if (Action == MPI_MASTER_RECV) {
					logMessagePID(slave, root, RECV_MATRIX_TAG);
					MPI_Recv(&(bufferMatrix[0][0]), sub_rows*sub_columns, MPI_DOUBLE, slave, MPI_MASTER_RECV, MPI_COMM_WORLD, &status);
					printMatrix(bufferMatrix, sub_rows, sub_columns);
					slideMatrix(Matrix, bufferMatrix, sub_rows, sub_columns, i * sub_rows, j * sub_columns, COMPOSE);
				}
			}
			slave++;
		}
	}
	free(bufferMatrix);
}

/*
 * Function:  slideMatrix
 * --------------------
 *	Given a Matrix it cuts into equal sizes and it places them into bufferMatrix
 *	Matrix:			Main Matrix to be cutted.
 *	subMatrix:		Destination matrix.
 *	sub_rows:		Sub matrix number of rows.
 *	sub_columns:	Sub matrix number of columns.
 *	row_offset:		Main matrix row offset.
 *	column_offset:	Main matrix column offset.
 *	choice:			0 to cut a Matrix from subMatrix
					1 to build a Matrix from subMatrix
 *	returns:		null
 */
void slideMatrix(double **Matrix, double **subMatrix, int sub_rows, int sub_columns, int row_offset, int column_offset, int choice)
{
	int i, j;

	for (i = 0; i < sub_rows; i++) {
		for (j = 0; j < sub_columns; j++) {
			if (choice == DECOMPOSE) {
				subMatrix[i][j] = Matrix[i + row_offset][j + column_offset];
			}
			else if (choice == COMPOSE) {
				Matrix[i + row_offset][j + column_offset] = subMatrix[i][j];
			}
			else {
				printf("Error wrong choice in decomposeMatrix function\n");
			}
		}
	}

}

/*
 * Function:  allocateMatrix
 * --------------------
 *	Allocates memory space to a given Matrix
 *	Matrix			Matrix to be created
 *	int rows:		Number of rows to be allocated in memory.
 *	int columns:	Number of columns to be allocated in memory.
 *  returns:		*Matrix
 */
double** allocateMatrix(int rows, int columns)
{
	double *data = (double *)malloc(rows*columns * sizeof(double));
	double **Matrix = (double **)malloc(rows * sizeof(double*));
	int i;
	for (i = 0; i < rows; i++)
	{
		Matrix[i] = &(data[columns*i]);
	}
	return Matrix;
}

/*
 * Function:  randomizeMatrix
 * --------------------
 *	Initiates a given Matrix generating always the same random numbers from srand
 *	** Matrix:	Matrix to be set to same random values.
 *	rows:		Number of rows of given matrix.
 *	columns:	Number of columns of given matrix.
 *  returns:		null
 */
void randomizeMatrix(double** Matrix, int rows, int columns)
{
	srand(1); // seed to generate always the same Matrix.
	int i, j;
	for (i = 0; i < rows; i++) {
		for (j = 0; j < columns; j++) {
			Matrix[i][j] = ((double)rand() / (double)(RAND_MAX)) + ((double)(rand() % 10)); //Generates float value between 0.0 and 10.0
		}
	}
}

/*
 * Function:  printMatrix
 * --------------------
 *	Prints a Matrix
 *	**Matrix:	Matrix to be printed.
 *	rows:		Number of rows of given matrix.
 *	columns:	Number of columns of given matrix.
 *	returns:	null
 */
void printMatrix(double** Matrix, int rows, int columns)
{
#if PRINT
	logMessagePID(processMPI, 0, PRINT_MATRIX_TAG);
	int i, j;
	printf("\n");
	for (i = 0; i < rows; i++) {
		printf("\t\t");
		for (j = 0; j < columns; j++) {
			printf("%.2e\t", Matrix[i][j]);
		}
		printf("\n");
	}
	printf("\n");
	fflush(stdout);
#endif
}

/*
 * Function:  printMatrixRealImag
 * --------------------
 *	Prints a Matrix and Imaginary
 *	**MatrixReal:	Matrix Real part to be printed.
  *	**MatrixImag:	Matrix Imaginary part to be printed.
 *	rows:		Number of rows of given matrix.
 *	columns:	Number of columns of given matrix.
 *	returns:		null
 */
void printMatrixRealImag(double** MatrixReal, double** MatrixImag, int rows, int columns)
{
#if PRINT
	logMessagePID(processMPI, 0, PRINT_MATRIX_TAG);
	int i, j;
	printf("\n");
	for (i = 0; i < rows; i++) {
		printf("\t\t");
		for (j = 0; j < columns; j++) {
			printf("%.2e + %.2e i\t", MatrixReal[i][j], MatrixImag[i][j]);
		}
		printf("\n");
	}
	printf("\n");
	fflush(stdout);
#endif
}

/*
 * Function:  Calculate subdimension
 * --------------------
 *	Given a n_columns, n_rows calculates how to divide Matrix on P number
 *	rows:			Main matrix rows.
 *	columns:		Main matrix coolumns.
 *	*sub_rows:		Calculated sub rows.
 *	*sub_columns:	Calculated sub columns.
 *	slides:			Number of chuncks matrix must be divided.
 *	returns:		null
 */
void calcMatrixSubDim(int rows, int columns, int *sub_rows, int *sub_columns, int n_slides) {
	int row_division = 2; //Start division by 2.
	int column_division = 2; //Start division by 2.

	while (rows % row_division != 0) {
		row_division++;
	}
	*sub_rows = rows / row_division;
	column_division = n_slides / row_division;
	*sub_columns = columns / column_division;
}

/*
 * Function:  logMessagePID
 * --------------------
 * Print different P_ID messages
 * P_IDsend:		Sender P_ID OR Main P_ID in use
 * P_IDrecv:		Receiver P_ID
 * MessageTag:		SEND_MATRIX_TAG
 *					RECV_MATRIX_TAG
 *					PRINT_MATRIX_TAG
 */
void logMessagePID(int P_IDsend, int P_IDrecv, int MessageTag)
{
#if LOGPID
	printf("%f\t", MPI_Wtime());
	switch (MessageTag) {
	case SEND_MATRIX_TAG:
		printf("P_ID%d sending to P_ID%d.", P_IDsend, P_IDrecv);
		break;
	case RECV_MATRIX_TAG:
		printf("P_ID%d receiving from P_ID%d.", P_IDrecv, P_IDsend);
		break;
	case PRINT_MATRIX_TAG:
		printf("P_ID%d displaying Matrix.", P_IDsend);
		break;
	case CALC_DFT_TAG:
		printf("P_ID%d calculating DFT.", P_IDsend);
		break;
	case INIT_TAG:
		printf("P_ID%d Initialized.", P_IDsend);
		break;
	}
	printf("\n");
	fflush(stdout);
#endif
}

/*
 * Function:  enteredMatrixDimCheck
 * --------------------
 * Check that entered Matrix Dimension is valid
 * *n_rows:		Entered n_rows
 * *n_columns:	Entered n_columns
 *	returns:	null
 */
void enteredMatrixDimCheck(int *n_rows, int *n_columns) {

	if (((*n_rows) * (*n_columns)) % numProcMPI != 0) {
		//printf("Adjusting rows to accomodate number of processes\n");
		*n_rows -= *n_rows % numProcMPI; // Ensures n_rows and n_columns is compatible with number of processes
		if (*n_columns % numProcMPI != 0) {
			//printf("Adjusting columns to accomodate number of processes\n");
			*n_columns -= *n_columns % numProcMPI;
		}
	}

	if (*n_rows == 0 || *n_columns == 0) {
		//printf("Non divisible number of rows or columns between computers %d.\n", numProcMPI);
		*n_rows = numProcMPI;
		*n_columns = numProcMPI;
	}
	//printf("Assigning values %d x %d to Main Matrix\n", *n_rows, *n_columns);

}
