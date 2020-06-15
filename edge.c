#include <png.h>
#include <zlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define file_name "input.png"       // Input file name
#define hist_file "histeql.pgm"     // Output file name for histogram equalized image
#define out_file "final.pgm"        // Output file name for image after edge detection

int **readImage(FILE *fp, int **rows, png_uint_32 *width, png_uint_32 *height, png_uint_32 *bit_depth, png_uint_32 *color_type) {
    /*
    * Read PNG image from external file
    * Supports Grayscale, Palette, Grayscale+Alpha, RGB and RGBA
    */
    png_structp pngptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop pnginfo = png_create_info_struct(pngptr);
    png_init_io(pngptr, fp);
    png_read_png(pngptr, pnginfo, PNG_TRANSFORM_IDENTITY, NULL);
    png_get_IHDR(pngptr, pnginfo, width, height, bit_depth, color_type, NULL, NULL, NULL);
    
    int color =  *color_type;

    rows = (int **) malloc(*height*sizeof(int));
    if (rows == NULL) {
        printf("Could not allocate memory to read image\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    png_bytepp rowP = png_get_rows(pngptr, pnginfo);
    int stride;

    if (color == 0 || color == 3) { // Grayscale or Palette
        stride = 1; 
    }
    else if (color == 2) {          // RGB
        stride = 3;
    }
    else if (color == 4) {          // Grayscale and Alpha
        stride == 2;
    }
    else if (color == 6) {          // RGB and Alpha
        stride = 4;
    }
    else {
        printf("Could not recognize color type\n");
        MPI_Abort(MPI_COMM_WORLD, 2);  
    }

    for(int i=0; i<*height; i++) {
        rows[i] = (int *) calloc(*width, sizeof(int));
        if (rows[i] == NULL) {
            printf("Could not allocate memory to read image\n");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        for(int j=0; j<*width; j++) {
            if (stride%2) {
                for(int k=0; k<stride; k++) {
                    rows[i][j] += rowP[i][j*stride+k];
                }
                rows[i][j] /= stride;
            }
            else {
                for(int k=0; k<stride-1; k++) {
                    rows[i][j] += rowP[i][j*stride+k];
                }
                rows[i][j] /= (stride-1);
            }
        }
    }

    return rows;
}

void writeImage(char *filename, int *sendbuff, unsigned int width, unsigned int height, unsigned int bit_depth, unsigned int color_type) {
    /*
    * Create .pgm image file
    * Supports only grayscale
    */
    FILE *fp = fopen(filename, "wb");
    if (fp == NULL) {
        printf("Failed to open output.pgm\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }
    fprintf(fp, "P2\n");
    fprintf(fp, "# %s\n", filename);
    fprintf(fp, "%u %d\n", width, height);
    fprintf(fp, "%u\n", (1<<bit_depth));

    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            if (j < width-1)
                fprintf(fp, "%d ", sendbuff[i*width + j]);
            else if (j == width-1)
                fprintf(fp, "%d\n", sendbuff[i*width + j]);
        }
    }
    fclose(fp);
}

void computeLocalHist(int *recvbuff, int localHist[], int newh, int width, int bit_depth) {
    /*
    * Compute the local histogram of the pixels present in recvbuff
    * Histogram is returned via localHist
    */
    for(int i=0; i<(1<<bit_depth); i++)
        localHist[i] = 0;
    for(int i=0; i<newh; i++) {
        for(int j=0; j<width; j++) {
            if (recvbuff[i*width + j] >= 0 && recvbuff[i*width + j] < (1<<bit_depth))
                localHist[recvbuff[i*width + j]]++;
            else if (recvbuff[i*width + j] < 0)
                localHist[0]++;
            else
                localHist[(1<<bit_depth)-1]++;
        }
    }
}

void computeCumulativeHist(int *localHist, int bit_depth) {
    /*
    * Replaces the histogram 'localHist' with the cumulative histogram
    */
    int sum = 0;
    for (int i=0; i<(1<<bit_depth); i++) {
        sum += localHist[i];
        localHist[i] = sum;
    }
}

int main(int argc, char *argv[]) {
    /*
    * Main function
    */
    
    int rank, size;

    MPI_Init(&argc, &argv);                 // Begin MPI environment

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        unsigned int data[4]; 
        int **rows = NULL;

        if (rank == 0) {
            FILE *fp = fopen(file_name, "rb");
            if (fp == NULL) {
                printf("Could not find file %s\n", file_name);
                MPI_Abort(MPI_COMM_WORLD, 0);
            }
            rows = readImage(fp, rows, &data[0], &data[1], &data[2], &data[3]);
            fclose(fp);
            if (rows == NULL) {
                printf("Failed to read image\n");
                MPI_Abort(MPI_COMM_WORLD, 0);
            }
        }
        MPI_Bcast(data, 4, MPI_INT, 0, MPI_COMM_WORLD);
        
        unsigned int height_original = data[1];
        unsigned int height = height_original + size - (height_original)%size;
        unsigned int width = data[0];
        unsigned int bit_depth = data[2];
        unsigned int color_type = data[3];
        
        unsigned int *sendbuff = (unsigned int *) malloc((height+1)*width*sizeof(unsigned int)); 
        if (sendbuff == NULL) {
            printf("Could not allocate memory for sending buffers\n");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }

        if (rank == 0) {
            for (int i=0; i<data[0]*data[1]; i++) {
                sendbuff[i] = (unsigned int) rows[i/data[0]][i%data[0]];
            }
            for (int i=data[0]*data[1]; i<height*width; i++) {
                sendbuff[i] = 0;
            }
        }

        int newh = height/size;
        
        int *recvbuff = (int *) malloc(newh*width*sizeof(int));
        if (recvbuff == NULL) {
            printf("Could not allocate memory for receiving buffers\n");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        MPI_Scatter(sendbuff, newh*width, MPI_INT, recvbuff, newh*width, MPI_INT, 0, MPI_COMM_WORLD);

        // --------------- Histogram equalization ------------------ //

        int *localHist = (int *) malloc((1<<bit_depth)*sizeof(int));
        int *localHistnew = (int *) malloc((1<<bit_depth)*sizeof(int));

        if (localHist == NULL || localHistnew == NULL) {
            printf("Could not allocate memory for storing histograms\n");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }

        if (rank < size-1)
            computeLocalHist(recvbuff, localHist, newh, width, bit_depth);
        else 
            computeLocalHist(recvbuff, localHist, newh-height+height_original, width, bit_depth);

        MPI_Allreduce(localHist, localHistnew, (1<<bit_depth), MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        computeCumulativeHist(localHistnew, bit_depth);

        for(int i=0; i<(1<<bit_depth); i++)
            localHist[i] = (localHistnew[i]*(1<<bit_depth))/(height_original*width);


        for(int i=0; i<newh; i++) {
            for(int j=0; j<width; j++) {
                if (recvbuff[i*width + j] < (1<<bit_depth) && recvbuff[i*width + j] > 0)
                    recvbuff[i*width + j] = localHist[recvbuff[i*width + j]];
                else if (recvbuff[i*width + j] < 0)
                    recvbuff[i*width + j] = localHist[0];
                else
                    recvbuff[i*width + j] = localHist[(1<<bit_depth)-1];
            }
        }

        // --------------- End of histogram equalization --------------------- //

        // -------------- Print Histogram equalized image ---- //

        MPI_Gather(recvbuff, newh*width, MPI_INT, sendbuff, newh*width, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            writeImage(hist_file, sendbuff, width, height_original, bit_depth, color_type);
        }

        // -------------- apply Sobel Filter ------------------ //

        int *upbuf = NULL;
        int *dwnbuf = NULL;
        MPI_Status status;

        if (size > 1) {
            if (rank%2 == 0) {
                if (rank == 0) {
                    upbuf = (int *) calloc(width, sizeof(int));
                    dwnbuf = (int *) malloc(width*sizeof(int));
                    //MPI_Send(pimage[newh-1], width, MPI_INT, 1, 1, MPI_COMM_WORLD);
                    MPI_Send(&recvbuff[(newh-1)*width], width, MPI_INT, 1, 1, MPI_COMM_WORLD);
                    MPI_Recv(dwnbuf, width, MPI_INT, 1, 0, MPI_COMM_WORLD, &status);
                }
                else if (rank == size-1) {
                    upbuf = (int *) malloc(width*sizeof(int));
                    dwnbuf = (int *) calloc(width, sizeof(int));
                    //MPI_Send(pimage[0], width, MPI_INT, rank-1, 0, MPI_COMM_WORLD);
                    MPI_Send(recvbuff, width, MPI_INT, rank-1, 0, MPI_COMM_WORLD);
                    MPI_Recv(upbuf, width, MPI_INT, rank-1, 1, MPI_COMM_WORLD, &status);
                }
                else {
                    upbuf = (int *) malloc(width*sizeof(int));
                    dwnbuf = (int *) malloc(width*sizeof(int));
                    //MPI_Send(pimage[0], width, MPI_INT, rank-1, 0, MPI_COMM_WORLD);
                    MPI_Send(recvbuff, width, MPI_INT, rank-1, 0, MPI_COMM_WORLD);
                    //MPI_Send(pimage[newh-1], width, MPI_INT, rank+1, 1, MPI_COMM_WORLD);
                    MPI_Send(&recvbuff[(newh-1)*width], width, MPI_INT, rank+1, 1, MPI_COMM_WORLD);
                    MPI_Recv(dwnbuf, width, MPI_INT, rank+1, 0, MPI_COMM_WORLD, &status);
                    MPI_Recv(upbuf, width, MPI_INT, rank-1, 1, MPI_COMM_WORLD, &status);
                }
            }
            else {
                if (rank == size-1) {
                    upbuf = (int *) malloc(width*sizeof(int));
                    dwnbuf = (int *) calloc(width, sizeof(int));
                    MPI_Recv(upbuf, width, MPI_INT, rank-1, 1, MPI_COMM_WORLD, &status);
                    //MPI_Send(pimage[0], width, MPI_INT, rank-1, 0, MPI_COMM_WORLD);
                    MPI_Send(recvbuff, width, MPI_INT, rank-1, 0, MPI_COMM_WORLD);
                }
                else {
                    upbuf = (int *) malloc(width*sizeof(int));
                    dwnbuf = (int *) malloc(width*sizeof(int));
                    MPI_Recv(dwnbuf, width, MPI_INT, rank+1, 0, MPI_COMM_WORLD, &status);
                    MPI_Recv(upbuf, width, MPI_INT, rank-1, 1, MPI_COMM_WORLD, &status);
                    //MPI_Send(pimage[0], width, MPI_INT, rank-1, 0, MPI_COMM_WORLD);
                    MPI_Send(recvbuff, width, MPI_INT, rank-1, 0, MPI_COMM_WORLD);
                    //MPI_Send(pimage[newh-1], width, MPI_INT, rank+1, 1, MPI_COMM_WORLD);
                    MPI_Send(&recvbuff[(newh-1)*width], width, MPI_INT, rank+1, 1, MPI_COMM_WORLD);
                }
            }
        }
        else {
            upbuf = (int *) calloc(width, sizeof(int));
            dwnbuf = (int *) calloc(width, sizeof(int));
        }

        int xfilter[3][3] = {{-1, 0, 1},{-2, 0, 2},{-1, 0, 1}};
        int yfilter[3][3] = {{1, 2, 1},{0, 0, 0},{-1, -2, -1}};

        int **outimage = (int **) malloc(newh*sizeof(int *));
        if (outimage == NULL) {
            printf("Failed to allocate memory for local output image\n");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        for (int i=0; i<newh; i++) {
            outimage[i] = (int *) malloc(width*sizeof(int));
            if (outimage[i] == NULL) {
                printf("Failed to allocate memory for local output image\n");
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
            for (int j=0; j<width; j++) {
                int magx = 0;
                int magy = 0;
                for (int k=0; k<3; k++) {
                    for (int l=0; l<3; l++) {
                        if ( (i+k-1 < newh && i+k-1 >= 0) && (j+l-1 < width && j+l-1 >=0 ) ) {
                            magx += xfilter[k][l]*recvbuff[(i+k-1)*width + j+l-1];
                            magy += yfilter[k][l]*recvbuff[(i+k-1)*width + j+l-1];
                        }
                        else if ( (i+k-1 < 0) && (j+l-1 < width && j+l-1 >=0 )) {
                            magx += xfilter[k][l]*upbuf[j+l-1];
                            magy += yfilter[k][l]*upbuf[j+l-1];
                        }
                        else if ( (i+k-1 >= newh) && (j+l-1 < width && j+l-1 >=0 )) {
                            magx += xfilter[k][l]*dwnbuf[j+l-1];
                            magy += yfilter[k][l]*dwnbuf[j+l-1];
                        }
                    }
                }
                if (magx >= (1<<bit_depth)) {
                    magx = (1<<bit_depth)-1;
                }
                else if (magx < 0) {
                    magx = 0;
                }
                if (magy >= (1<<bit_depth)) {
                    magy = (1<<bit_depth)-1;
                }
                else if (magy < 0) {
                    magy = 0;
                }
                outimage[i][j] = (int) sqrt(magx*magx + magy*magy);
            }
        }

        // ---------------- Sobel filter ends -------------------- //

        // -------------- Print Output ------------------- //

        for(int i=0; i<newh; i++) {
            for(int j=0; j<width; j++)
                recvbuff[i*width + j] = outimage[i][j] ;
        }

        MPI_Gather(recvbuff, newh*width, MPI_INT, sendbuff, newh*width, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            writeImage(out_file, sendbuff, width, height_original, bit_depth, color_type);
        }

    MPI_Finalize();

    return 0;
}