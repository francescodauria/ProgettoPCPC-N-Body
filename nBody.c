#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define SOFTENING 1e-9f

typedef struct
{
    float x, y, z;
} pBody;
typedef struct
{
    float vx, vy, vz;
} vBody;

void randomizeBodies(float *data, int n)
{
    for (int i = 0; i < n; i++)
    {
        data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}

void printPointsArray(int rank, pBody *array, int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%d:[%f,%f,%f]\n", rank, array[i].x, array[i].y, array[i].z);
    }
}

void printVelocityArray(int rank, vBody *array, int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%d:[%f,%f,%f]\n", rank, array[i].vx, array[i].vy, array[i].vz);
    }
}


void bodyForce(pBody *p, vBody *v, float dt, int start, int end, int myStart, int myEnd)
{
    for (int i = myStart; i < myEnd; i++)
    {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;

        for (int j = start; j < end; j++)
        {
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = 1.0f / sqrt(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        v[i].vx += dt * Fx;
        v[i].vy += dt * Fy;
        v[i].vz += dt * Fz;
    }
}

void calculateStartEnd(int nBodies, int sizeProc, int *startIndexes, int *endIndexes)
{
    int start;
    int end;
    int resto = nBodies % sizeProc;
    int sum = 0;
    for (int i = 0; i < sizeProc; i++)
    {
        startIndexes[i] = sum;
        if (resto > 0)
        {
            end = nBodies / sizeProc + 1;
            resto--;
        }
        else
            end = nBodies / sizeProc;

        sum += --end;
        endIndexes[i] = sum + 1;
        sum++;
    }
}

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int nBodies = 10;
    int iterations = 1;
    int bcast;
    if (argc != 4)
    {
        if (rank == 0)
            printf("****Usage mpirun -np X nBody [-b|-s] N I****\n");
        MPI_Finalize();
        return 0;
    }
    if (strcmp(argv[1], "-b") == 0)
        bcast = 1;
    else if (strcmp(argv[1], "-s") == 0)
        bcast = 0;
    else
    {
        if (rank == 0)
            printf("****Usage mpirun -np X nBody [-b|-s] N I****\n");
        MPI_Finalize();
        return 0;
    }
    if (argc > 2)
        nBodies = atoi(argv[2]);
    if (argc > 3)
        iterations = atoi(argv[3]);
    srand(42);
    const float dt = 0.01f; // time step

    int bytes = nBodies * sizeof(vBody);
    float *buf = (float *)malloc(bytes);
    float *buf2 = (float *)malloc(bytes);
    vBody *v = (vBody *)buf;
    pBody *p = (pBody *)buf2;

    int *sendcounts;
    // array describing how many elements to send to each process
    int *displs;
    // array describing the displacements where each segment begins
    int rem = nBodies % size;
    // elements remaining after division among processes
    int sum = 0;
    // Sum of counts. Used to calculate displacements

    MPI_Request requestBp = MPI_REQUEST_NULL;
    //request points broadcast
    MPI_Request requestBv = MPI_REQUEST_NULL;
    //request velocity broadcast
    float timeStart, timeEnd;

    if (rank == 0)
    {
        randomizeBodies(buf, 3 * nBodies);  // Init vel data
        randomizeBodies(buf2, 3 * nBodies); // Init pos data
    }
    MPI_Datatype bodies_datatype, old_types[1];
    int blockcounts[1];
    MPI_Aint offset[1];
    offset[0] = 0;
    old_types[0] = MPI_FLOAT;
    blockcounts[0] = 3;
    MPI_Type_create_struct(1, blockcounts, offset, old_types, &bodies_datatype);
    MPI_Type_commit(&bodies_datatype);
        //commit datatype
    int *startIndexes = (int *)malloc(size * sizeof(int));
    int *endIndexes = (int *)malloc(size * sizeof(int));
    calculateStartEnd(nBodies, size, startIndexes, endIndexes);
    if (rank == 0)
    {
        printf("Original points array\n");
        printPointsArray(rank, p, nBodies);
        printf("Original velocity array\n");
        printVelocityArray(rank, v, nBodies);
        printf("\n");
    }
    sendcounts = malloc(sizeof(int) * size);
    displs = malloc(sizeof(int) * size);
   
    for (int i = 0; i < size; i++)
    {
        sendcounts[i] = nBodies / size;
        if (rem > 0)
        {
            sendcounts[i] += 1;
            rem--;
        }
        displs[i] = sum;
        sum += sendcounts[i];
    }
    MPI_Barrier(MPI_COMM_WORLD);
    timeStart = MPI_Wtime();
    MPI_Scatterv(v, sendcounts, displs, bodies_datatype, &v[startIndexes[rank]], sendcounts[rank], bodies_datatype, 0, MPI_COMM_WORLD);
    MPI_Scatterv(p, sendcounts, displs, bodies_datatype, &p[startIndexes[rank]], sendcounts[rank], bodies_datatype, 0, MPI_COMM_WORLD);
        //send point velocity and positions
    MPI_Request requests[size];
    MPI_Status stats;

    for (int ite = 0; ite < iterations; ite++)
    {
        if (bcast == 1)
        {
            for (int root = 0; root < size; ++root)
            {
                if (root == rank)
                {
                    MPI_Ibcast(&p[startIndexes[rank]], sendcounts[rank], bodies_datatype, rank, MPI_COMM_WORLD, &requests[rank]);
                        //non blocking send broadcasting
                }
                else
                {
                    MPI_Ibcast(&p[startIndexes[root]], sendcounts[root], bodies_datatype, root, MPI_COMM_WORLD, &requests[root]);
                        //non blocking receive broadcasting
                }
            }
        }
        else
        {
            for (int i = 0; i < size; i++) 
            {

                if (i != rank)
                {
                    MPI_Isend(&p[startIndexes[rank]], sendcounts[rank], bodies_datatype, i, 0, MPI_COMM_WORLD, &requestBp);
                        //non blocking send
                    MPI_Irecv(&p[startIndexes[i]], sendcounts[i], bodies_datatype, i, 0, MPI_COMM_WORLD, &requests[i]);
                        //non blocking receive
                }
            }
        }
        bodyForce(p, v, dt, startIndexes[rank], endIndexes[rank], startIndexes[rank], endIndexes[rank]);
            //compute bodyforce on my particles
        for (int j = 0; j < size; j++) //to order the comparisons
        {
            if (j != rank)
            {
                MPI_Wait(&requests[j], &stats);
                    //wait the process with rank = j
                bodyForce(p, v, dt, startIndexes[j], endIndexes[j], startIndexes[rank], endIndexes[rank]);
                    //compute bodyforce on my particles comparing with process with rank = j particles
            }
        }
        for (int i = startIndexes[rank]; i < endIndexes[rank]; i++)
        { // integrate position
            p[i].x += v[i].vx * dt;
            p[i].y += v[i].vy * dt;
            p[i].z += v[i].vz * dt;
        }
    }
    MPI_Gatherv(&p[startIndexes[rank]], sendcounts[rank], bodies_datatype, p, sendcounts, displs, bodies_datatype, 0, MPI_COMM_WORLD);
        //gather to process with rank = 0 particles positions
    MPI_Gatherv(&v[startIndexes[rank]], sendcounts[rank], bodies_datatype, v, sendcounts, displs, bodies_datatype, 0, MPI_COMM_WORLD);
            //gather to process with rank = 0 particles velocity
    MPI_Barrier(MPI_COMM_WORLD);
    timeEnd = MPI_Wtime();
    if (rank == 0)
        printf("Execution time %lf s\n\n", timeEnd - timeStart);
    if (rank == 0)
    {
        printf("Points array\n");
        printPointsArray(rank, p, nBodies);
        printf("Velocity array\n");
        printVelocityArray(rank, v, nBodies);
        printf("\n");
    }
    MPI_Type_free(&bodies_datatype);
        //free datatype
    MPI_Finalize();
}