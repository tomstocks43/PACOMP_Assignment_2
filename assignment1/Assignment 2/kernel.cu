#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector_types.h>
#include <vector_functions.h>
#include "cuda_texture_types.h"
#include "texture_fetch_functions.hpp"


#define IMAGE_DIM 2048
#define SAMPLE_SIZE 6
#define NUMBER_OF_SAMPLES (((SAMPLE_SIZE*2)+1)*((SAMPLE_SIZE*2)+1))
#define FAILURE 0
#define SUCCESS !FAILURE
#define _CRT_SECURE_NO_WARNINGS
#define HEADCHAR 24
#define USER_NAME "acp18ts"		//replace with your user name

#define CUDA_CALL(x) {cudaError_t cuda_error__ = (x); if (cuda_error__) printf("CUDA error: " #x " returned \"%s\"\n", cudaGetErrorString(cuda_error__));}
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define INF     2e10f


typedef struct {
	unsigned char r, g, b;
} Pixel;

typedef struct {
	int r, g, b;
} Pixel2;

typedef struct {
	unsigned int height, width, magic, max_val;
	Pixel2 *data;
} Image;

typedef struct {
	int start, stop;
} Timer;
typedef enum MODE { CPU, OPENMP, CUDA, ALL } MODE;

void print_help();
int process_command_line(int argc, char *argv[]);
static Image *import_ppm_hdr(char *imagein, char *binmode);
int powerof2(unsigned int c);
static Image *pad_array(Image *im, unsigned int c, Pixel2 *glob_av);
static Image *seq_mosaic_filter(Image *im, unsigned int c);
void output_ppm(char *filename, Image *im);
static Image *omp_mosaic_filter(Image *im, unsigned int c);
static Pixel2 *seq_print_av_col(Image *im);
static Pixel2 *omp_print_av_col(Image *im);
void get_file_dims(const char* filename, int *w, int *h);
void output_image_file(uchar4* image, const char* image_out, int w, int h);
void input_image_file(const char* filename, uchar4* image);


inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

/* DEVICE CODE*///##########################################################################################

__global__ void cuda_simple_mosaic_kern(Image *im, int *numel) {
	// secify variables we need for calculating indexes
	int skipline = im->width;
	int mosaic_start = (threadIdx.x*blockDim.x) + (threadIdx.y*blockDim.y*skipline);
	int relative_index = 0, absolute_index = 0;
	int i, j;

	int tile_av_r = 0, tile_av_g = 0, tile_av_b = 0;

	// iterate through mosaic accumulating values
	for (i = 0; i >= blockDim.x; i++) {
		for (j = 0; j >= blockDim.y; j++) {
			relative_index = i + j * skipline;
			absolute_index = mosaic_start + relative_index;
			if (absolute_index > *numel) {
				tile_av_r += im->data[absolute_index].r;
				tile_av_g += im->data[absolute_index].g;
				tile_av_b += im->data[absolute_index].b;
			}
		}
	}

	// calculate mean of all R, G and B values
	tile_av_r = (int)round((double)tile_av_r / (double)(blockDim.x*blockDim.y));
	tile_av_g = (int)round((double)tile_av_g / (double)(blockDim.x*blockDim.y));
	tile_av_b = (int)round((double)tile_av_b / (double)(blockDim.x*blockDim.y));

	// reiterate and assign averages to pixels.
	for (i = 0; i >= blockDim.x; i++) {
		for (j = 0; j >= blockDim.y; j++) {
			relative_index = i + j * skipline;
			absolute_index = mosaic_start + relative_index;
			if (absolute_index > *numel) {
				im->data[absolute_index].r = tile_av_r;
				im->data[absolute_index].g = tile_av_g;
				im->data[absolute_index].b = tile_av_b;
			}
		}
	}
}

__global__ void cuda_inter_mosaic_kern(Image *im, Pixel2 *data, Pixel2 *tile_averages) {
	// secify variables we need for calculating indexes
	int skipline = im->width;
	int mosaic_start = (blockIdx.x*blockDim.x) + (blockIdx.y*blockDim.y*skipline);
	int thread_relative = threadIdx.x + (threadIdx.y*skipline);
	int idx = mosaic_start + thread_relative;
	int tile_avs_idx = blockIdx.x + (blockIdx.y*gridDim.x);

	int truey = blockIdx.y*blockDim.y + threadIdx.y;
	int truex = blockIdx.x*blockDim.x + threadIdx.x;

	int tempr = 0, tempg = 0, tempb = 0;

	if (truey < im->height && truex < im->width) {
		tempr = data[idx].r;
		tempg = data[idx].g;
		tempb = data[idx].b;

		int *r, *g, *b;
		r = &tile_averages[tile_avs_idx].r;
		g = &tile_averages[tile_avs_idx].g;
		b = &tile_averages[tile_avs_idx].b;

		atomicAdd(r, tempr);
		atomicAdd(g, tempg);
		atomicAdd(b, tempb);



	}
	__syncthreads();
	if (truey < im->height && truex < im->width) {
		data[idx] = tile_averages[tile_avs_idx];
	}

}

__global__ void image_blur(uchar4 *image, uchar4 *image_output, int c, int w, int h) {
	// map from threadIdx/BlockIdx to pixel position
	int skipline = w;
	int idx = threadIdx.x+(blockIdx.x*blockDim.x) + (threadIdx.y*skipline)+(blockIdx.y*blockDim.y*skipline);
	extern __shared__ int4 tile_av;

	int truey = blockIdx.y*blockDim.y + threadIdx.y;
	int truex = blockIdx.x*blockDim.x + threadIdx.x;
	double numel = c * c;

	if (threadIdx.x+threadIdx.y == 0) {
		tile_av.x = 0;
		tile_av.y = 0;
		tile_av.z = 0;
	}
	__syncthreads();

	if (truey < h && truex < w) {
		int *addr_r, *addr_g, *addr_b;
		int localr, localg, localb;

		localr = (int)image[idx].x;
		localg = (int)image[idx].y;
		localb = (int)image[idx].z;

		addr_r = &tile_av.x;
		addr_g = &tile_av.y;
		addr_b = &tile_av.z;

		atomicAdd(addr_r, localr);
		atomicAdd(addr_g, localg);
		atomicAdd(addr_b, localb);
	}
	__syncthreads();
	if (truey < h && truex < w) {
		image_output[idx].x = (unsigned char)(round((double)tile_av.x / numel));
		image_output[idx].y = (unsigned char)(round((double)tile_av.y / numel));
		image_output[idx].z = (unsigned char)(round((double)tile_av.z / numel));
	}

}

__global__ void image_av_col(uchar4 *image, int w, int h, int4 *av) {
	// map from threadIdx/BlockIdx to pixel position
	int skipline = w;
	int block_start_index = (blockIdx.x*blockDim.x) + (blockIdx.y*blockDim.y*skipline);
	int relative_index = threadIdx.x + (threadIdx.y*skipline);
	int idx = block_start_index + relative_index;
	
	int truey = blockIdx.y*blockDim.y + threadIdx.y;
	int truex = blockIdx.x*blockDim.x + threadIdx.x;

	__syncthreads();

	if (idx < w*h) {
		int *addr_r, *addr_g, *addr_b;
		int localr, localg, localb;

		localr = image[idx].x;
		localg = image[idx].y;
		localb = image[idx].z;

		addr_r = &av->x;
		addr_g = &av->y;
		addr_b = &av->z;

		atomicAdd(addr_r, localr);
		atomicAdd(addr_g, localg);
		atomicAdd(addr_b, localb);
		if (idx == 512*512) {
			printf("Thread is %d %d, %d is being added \n", truex, truey, localr);
		}
	}

	__syncthreads();
	if (idx == 0) {
		int avr, avg, avb;
		avr = (int)round((double)av->x / (double)(w*h));
		avg = (int)round((double)av->y / (double)(w*h));
		avb = (int)round((double)av->z / (double)(w*h));

		printf("CUDA Average image colour red = %d, green = %d, blue = %d \n", avr, avg, avb);
	}

}

/* HOST CODE*///###########################################################################################

unsigned int c = 0;
MODE execution_mode = CPU;

char *mode;
char *imagein;
char *imageout;
char *binmode;

int main(int argc, char *argv[]) {

	// Checking command line inputs for user error
	if (process_command_line(argc, argv) == FAILURE)
		return 1;
	if (powerof2(c) != 1) {
		printf("%d is not a power of 2!", c);
		exit(2);
	}



	//TODO: execute the mosaic filter based on the mode
	if (strcmp(mode, "CPU") == 0) {
		execution_mode = CPU;
	}
	else if (strcmp(mode, "OPENMP") == 0) {
		execution_mode = OPENMP;
	}
	else if (strcmp(mode, "CUDA") == 0) {
		execution_mode = CUDA;
	}
	else if (strcmp(mode, "ALL") == 0) {
		execution_mode = ALL;

	}
	else {
		printf("execution mode %s is not recognised.", mode);
		exit(11);
	}
	switch (execution_mode) {
	case (CPU): {

	
		//TODO: read input image file (either binary or plain text PPM) 
		Image *im_in;
		im_in = (Image *)malloc(sizeof(Image));
		im_in = import_ppm_hdr(imagein, binmode);

		// checking C is not bigger than image dims
		if (c > im_in->height || c > im_in->width) {
			printf("%d is bigger than image dimensions of %d by %d", c, im_in->height, im_in->width);
			exit(3);
		}


		clock_t start, stop;
		start = clock();
		Pixel2 *glob_av;
		glob_av = (Pixel2*)malloc(sizeof(Pixel2));
		glob_av = seq_print_av_col(im_in);
		//Start the mosaic filter
		im_in = pad_array(im_in, c, glob_av);
		im_in = seq_mosaic_filter(im_in, c);
		//TODO: end timing here
		stop = clock();
		printf("CPU mode execution time took %.4fs\n", ((double)(stop - start) / CLOCKS_PER_SEC));

		output_ppm(imageout, im_in);

		break;
	}
	case (OPENMP): {

		//TODO: read input image file (either binary or plain text PPM) 
		Image *im_in;
		im_in = (Image *)malloc(sizeof(Image));
		im_in = import_ppm_hdr(imagein, binmode);

		// checking C is not bigger than image dims
		if (c > im_in->height || c > im_in->width) {
			printf("%d is bigger than image dimensions of %d by %d", c, im_in->height, im_in->width);
			exit(3);
		}


		//TODO: starting timing here
		clock_t start, stop;
		start = clock();
		//TODO: calculate the average colour value
		Pixel2 *glob_av;
		glob_av = (Pixel2*)malloc(sizeof(Pixel2));
		glob_av = omp_print_av_col(im_in);
		// Output the average colour value for the image
		im_in = pad_array(im_in, c, glob_av);
		im_in = omp_mosaic_filter(im_in, c);
		//TODO: end timing here
		stop = clock();
		printf("OPENMP mode execution time took %.4fs\n", ((double)(stop - start) / CLOCKS_PER_SEC));

		output_ppm(imageout, im_in);

		break;
	}
	case (CUDA): {

		unsigned int image_size;
		uchar4 *d_image, *d_image_output;
		uchar4 *h_image;
		cudaEvent_t start, stop;
		float ms; 

		// create timers
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		// allocate and load host image
		int w, h, *d_c;
		get_file_dims(imagein, &w, &h);
		image_size = w * h * sizeof(uchar4);

		h_image = (uchar4*)malloc(w * h * sizeof(uchar4));
		input_image_file(imagein, h_image);

		gpuErrchk(cudaHostRegister(h_image, w * h * sizeof(uchar4), cudaHostRegisterDefault));

		// allocate memory on the GPU for the image and output image
		gpuErrchk(cudaMalloc((void**)&d_image, image_size));
		gpuErrchk(cudaMalloc((void**)&d_image_output, image_size));

		

		// copy image to device memory
		gpuErrchk(cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice));


		int num_tile_y, num_tile_x;
		num_tile_y = (int)ceil((double)h / (double)32);
		num_tile_x = (int)ceil((double)w / (double)32);

		//cuda layout (print average)
		dim3    blocksPerGrid_av(num_tile_x, num_tile_y);
		dim3    threadsPerBlock_av(32, 32);

		// normal version (print average)
		cudaEventRecord(start, 0);

		int4 *d_global_av;
		gpuErrchk(cudaMalloc((void**)&d_global_av, sizeof(int4)));
		gpuErrchk(cudaMemset(d_global_av, 0, sizeof(int4)));

		image_av_col <<<blocksPerGrid_av, threadsPerBlock_av >>>(d_image, w, h, d_global_av);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());


		num_tile_y = (int)ceil((double)h / (double)c);
		num_tile_x = (int)ceil((double)w / (double)c);


		if (c <= 32) {
			//cuda layout
			dim3    blocksPerGrid(num_tile_x, num_tile_y);
			dim3    threadsPerBlock(c, c);

			// normal version
			cudaEventRecord(start, 0);
			image_blur << <blocksPerGrid, threadsPerBlock >> > (d_image, d_image_output, c, w, h);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());

		}
		else if (c > 32) {
			printf("This program uses 1 thread block per mosaic so mosaic sizes > 1024 are not supported");
			exit(1);
		}

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&ms, start, stop);

		// copy the image back from the GPU for output to file
		gpuErrchk(cudaMemcpy(h_image, d_image_output, image_size, cudaMemcpyDeviceToHost));

		// output image
		printf("Outputting file...");
		output_image_file(h_image, imageout, w, h);

		//cleanup
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		cudaFree(d_image);
		cudaFree(d_image_output);
		free(h_image);
		cudaFree(d_global_av);

		printf("CUDA mode execution time took %.4fs\n", ((ms) / CLOCKS_PER_SEC));
		break;
	}
	case (ALL): {
		//TODO: starting timing here
		// Checking command line inputs for user error
		
		//TODO: read input image file (either binary or plain text PPM) 
		Image *im_in;
		im_in = (Image *)malloc(sizeof(Image));
		im_in = import_ppm_hdr(imagein, binmode);

		// checking C is not bigger than image dims
		if (c > im_in->height || c > im_in->width) {
			printf("%d is bigger than image dimensions of %d by %d", c, im_in->height, im_in->width);
			exit(3);
		}
		clock_t start, stop;
		start = clock();
		Pixel2 *glob_av;
		glob_av = (Pixel2*)malloc(sizeof(Pixel2));
		glob_av = seq_print_av_col(im_in);
		//Start the mosaic filter
		im_in = pad_array(im_in, c, glob_av);
		im_in = seq_mosaic_filter(im_in, c);
		//TODO: end timing here
		stop = clock();
		printf("CPU mode execution time took %.4fs\n", ((double)(stop - start) / CLOCKS_PER_SEC));


		break;
	}
	}

	//save the output image file (from last executed mode)
	cudaDeviceReset();
	return 0;
}

void print_help() {
	printf("mosaic_%s C M -i input_file -o output_file [options]\n", USER_NAME);

	printf("where:\n");
	printf("\tC              Is the mosaic cell size which should be any positive\n"
		"\t               power of 2 number \n");
	printf("\tM              Is the mode with a value of either CPU, OPENMP, CUDA or\n"
		"\t               ALL. The mode specifies which version of the simulation\n"
		"\t               code should execute. ALL should execute each mode in\n"
		"\t               turn.\n");
	printf("\t-i input_file  Specifies an input image file\n");
	printf("\t-o output_file Specifies an output image file which will be used\n"
		"\t               to write the mosaic image\n");
	printf("[options]:\n");
	printf("\t-f ppm_format  PPM image output format either PPM_BINARY (default) or \n"
		"\t               PPM_PLAIN_TEXT\n ");
}

int process_command_line(int argc, char *argv[]) {
	if (argc < 7) {
		fprintf(stderr, "Error: Missing program arguments. Correct usage is...\n");
		print_help();
		return FAILURE;
	}

	//first argument is always the executable name

	//read in the non optional command line arguments
	c = (unsigned int)atoi(argv[1]);

	//TODO: read in the mode
	int _argc = 2;
	mode = argv[_argc++];
	printf("C is %d \n", c);
	printf("mode is: %s\n", mode);
	//Validate mode

	imageout = 0; // SET DEFUALTS
	imagein = 0;
	binmode = (char*)"PPM_BINARY";
	while (_argc < argc)
	{
		if (strcmp("-o", argv[_argc]) == 0) //WHEN -0 IS FOUND IN ARGS SET NEXT ARG TO OUPUT
		{
			imageout = argv[++_argc];
			printf("imageout: %s\n", imageout);
		}
		else if (strcmp("-i", argv[_argc]) == 0) //WHEN -I IS FOUND IN ARGS SET NEXT ARG TO INPUT
		{
			imagein = argv[++_argc];
			printf("imagein: %s\n", imagein);
		}
		else if (strcmp("-f", argv[_argc]) == 0) //WHEN -F IS FOUND SET NEXT ARG TO BINARY MODE
		{
			binmode = argv[++_argc];
			printf("binary mode: %s\n", binmode);
		}
		else
		{
			printf("Unexpected arg: %s\n", argv[_argc]);
			return FAILURE;
		}
		++_argc;
	}
	if (imageout == 0 || imagein == 0)
		return FAILURE;
	return SUCCESS;
}

static Image *import_ppm_hdr(char *imagein, char *binmode) {


	Image *im; // define image struct called img
	FILE *file_point;
	char magic[64]; // define 64 long array for first line of header, 64 incase there is a long comment before first \n
	char temp[64];
	int width, height, max_val;
	int i = 0;

	if (strcmp(binmode, "PPM_BINARY") == 0) {
		file_point = fopen(imagein, "rb");
		if (file_point == NULL) {
			printf("Could not open file %s", imagein);
			exit(4);
		}
	}
	else if (strcmp(binmode, "PPM_PLAIN_TEXT") == 0) {
		file_point = fopen(imagein, "r");
		if (file_point == NULL) {
			printf("Could not open file %s", imagein);
			exit(4);
		}
	}
	else {
		printf("%s is not a valid PPM mode", binmode);
		exit(10);
	}


	fgets(magic, sizeof(magic), file_point);
	while (magic[0] != 'P' && i < 20) {
		fgets(magic, sizeof(magic), file_point);
		i++;
	}
	if (magic[0] != 'P') {
		printf("%s is not a PPM file or there are too many comments before header data", imagein);
		exit(5);
	}

	//check the image format
	if (magic[1] != '3' && magic[1] != '6') { // makes sure first line is magic num p6 for bin, p9 for ascii
		fprintf(stderr, "Image format must be 'P6' or 'P3\n");
		printf("magic num is %c %c", magic[0], magic[1]);
		exit(6);
	}

	im = (Image *)malloc(sizeof(Image));
	//skip any potential comments 
	fgets(temp, sizeof(temp), file_point);
	while (temp[0] == '#') {
		fgets(temp, sizeof(temp), file_point);
	}
	sscanf(temp, "%d", &width); // assign next found number to width

								//skip any potential comments 
	fgets(temp, sizeof(temp), file_point);
	while (temp[0] == '#') {
		fgets(temp, sizeof(temp), file_point);
	}
	sscanf(temp, "%d", &height); // assign next found number to height

								 //skip any potential comments 
	fgets(temp, sizeof(temp), file_point);
	while (temp[0] == '#') {
		fgets(temp, sizeof(temp), file_point);
	}
	sscanf(temp, "%d \n", &max_val); // assign next found number to max

								  //while (fgetc(file_point) != '\n');

	Pixel* data;
	data = (Pixel*)malloc(width * height * sizeof(Pixel)); //memory needed for image = number of elements * memory size of 1 pixel
	Pixel2* data_int;
	data_int = (Pixel2*)malloc(width * height * sizeof(Pixel2));

	if (!im) { // check if mem allocation fails
		printf("Unable to allocate memory for image data \n");
		exit(7);
	}

	//read pixel data from file
	if (strcmp(binmode, "PPM_BINARY") == 0) {
		if (fread(data, sizeof(Pixel), width*height, file_point) != width * height) { // reads into img data, the number of elements to be read is 3 (rgb) * number of x elements because each element in the array is a y row. Thus third arg is the size of a y row.
			fprintf(stderr, "Error loading image '%s'\n", imagein);
			exit(8);
		}
	}

	else if (strcmp(binmode, "PPM_PLAIN_TEXT") == 0) {
		for (i = 0; i < width * height; i++) {
			fscanf(file_point, "%c %c %c", &data[i].r, &data[i].g, &data[i].b);
		}
	}


	int numel = width * height;

	for (i = 0; i < numel; i++) {
		data_int[i].r = data[i].r;
		data_int[i].g = data[i].g;
		data_int[i].b = data[i].b;
	}

	im->magic = magic[1] - '0';
	im->height = height;
	im->width = width;
	im->max_val = max_val;
	im->data = data_int;

	fclose(file_point);
	return im;
}

int powerof2(unsigned int c)
{
	while (((c % 2) == 0) && c > 1)
		c /= 2;
	return (c == 1);
}

static Image *pad_array(Image *im, unsigned int c, Pixel2 *glob_av) {

	unsigned int rows_required, cols_required;
	rows_required = (unsigned int)ceil((double)im->height / c);
	cols_required = (unsigned int)ceil((double)im->width / c);
	unsigned int windows_width = cols_required * c;
	unsigned int windows_height = rows_required * c;
	int pad_sides = (int)ceil((float)(windows_width - im->width) / 2);
	int pad_height = (int)ceil((float)(windows_height - im->height) / 2);

	if (pad_height != 0 || pad_sides != 0) {
		Pixel2* padded;
		padded = (Pixel2*)malloc(windows_width * windows_height * sizeof(Pixel2));
		unsigned int i, j;
		for (i = 0; i < windows_width * windows_height; i++) {
			padded[i] = *glob_av;
		}
		int idx, idx_pad;
		for (i = 0; i < im->width; i++) {
			for (j = 0; j < im->height; j++) {
				idx = i * im->width + j;
				idx_pad = (i + pad_sides)*(im->width + pad_sides) + (j + pad_height);
				padded[idx_pad] = im->data[idx];


			}
		}
		im->data = padded;
		im->height = windows_height;
		im->width = 1337;
	}
	return(im);
}

static Image *seq_mosaic_filter(Image *im, unsigned int c) {
	// need to find indices of where mosiac windows will be applied so we can iterate
	unsigned int rows_required, cols_required;
	rows_required = im->height / c;
	cols_required = im->width / c;
	int *height_idxs, *width_idxs;
	unsigned int i = 0, j = 0, k = 0, l = 0;
	height_idxs = (int*)malloc(rows_required * sizeof(int));
	width_idxs = (int*)malloc(cols_required * sizeof(int));
	for (i = 0; i < rows_required; i++) {
		height_idxs[i] = c * i;
	}
	for (i = 0; i < cols_required; i++) {
		width_idxs[i] = c * i;

	}

	int *window_idxs;
	window_idxs = (int*)malloc(c * c * sizeof(int));

	int col_idx, row_idx, im_idx, relative_index, write_idx;
	double local_r = 0, local_g = 0, local_b = 0, c_dub = (double)c;
	Pixel2 *local_av;
	local_av = (Pixel2*)malloc(sizeof(Pixel2));
	local_av->r = 0; local_av->g = 0; local_av->b = 0;

	for (i = 0; i < cols_required; i++) {
		col_idx = i * c * im->width; //i*c to skip to next window index and then * width to account for flattened array
		for (j = 0; j < rows_required; j++) {

			row_idx = (j * c);
			im_idx = col_idx + row_idx;
			// iterating through idx of every top right corner of windows
			for (k = 0; k < c; k++) {
				for (l = 0; l < c; l++) {
					relative_index = (k*im->width) + l;
					window_idxs[k*c + l] = im_idx + relative_index;
					local_r = local_r + (double)im->data[im_idx + (k * im->width) + l].r;
					local_g = local_g + (double)im->data[im_idx + (k * im->width) + l].g;
					local_b = local_b + (double)im->data[im_idx + (k * im->width) + l].b;
				}
			}
			local_r = local_r / (c_dub*c_dub);
			local_g = local_g / (c_dub*c_dub);
			local_b = local_b / (c_dub*c_dub);

			local_av->r = (int)round(local_r);
			local_av->g = (int)round(local_g);
			local_av->b = (int)round(local_b);
			local_r = 0, local_g = 0, local_b = 0;
			for (k = 0; k < c*c; k++) { // iterate through the indexes we stored when we were calculating the window average and assign averge values into the image.
				write_idx = window_idxs[k];
				im->data[write_idx] = *local_av;
			}

		}
	}
	return(im);
}

void output_ppm(char *filename, Image *im) {
	FILE *file_point;
	//open file for output
	file_point = fopen(filename, "w");

	if (!file_point) {
		fprintf(stderr, "Unable to open file '%s'\n", filename);
		exit(12);
	}

	//write the header file
	//image format
	fprintf(file_point, "P6\n");


	//image size
	fprintf(file_point, "%d\n%d\n", im->width, im->height);

	// rgb component depth
	fprintf(file_point, "%d\n", im->max_val);

	Pixel *char_data;
	char_data = (Pixel*)malloc(im->width * im->height * sizeof(Pixel)); //memory needed for image = number of elements * memory size of 1 pixel
	for (unsigned int i = 0; i < im->height*im->width; i++) {
		char_data[i].r = (unsigned char)im->data[i].r;
		char_data[i].g = (unsigned char)im->data[i].g;
		char_data[i].b = (unsigned char)im->data[i].b;
	}

	// pixel data
	/*unsigned int i;
	for (i = 0; i < im->width * im->height; i++) {
	fprintf(file_point, "%c %c %c ", char_data[i].r, char_data[i].g, char_data[i].b);
	}*/

	fwrite(char_data, sizeof(Pixel), im->height*im->width, file_point);
	fclose(file_point);
}

static Image *omp_mosaic_filter(Image *im, unsigned int c) {
	// need to find indices of where mosiac windows will be applied so we can iterate
	int rows_required, cols_required;
	rows_required = im->height / c;
	cols_required = im->width / c;
	int *height_idxs, *width_idxs;
	height_idxs = (int*)malloc(rows_required * sizeof(int));
	width_idxs = (int*)malloc(cols_required * sizeof(int));
	int n;
	double c_2 = (double)c*c;


	for (n = 0; n < rows_required; n++) {
		height_idxs[n] = c * n;
	}
	for (n = 0; n < cols_required; n++) {
		width_idxs[n] = c * n;

	}

#pragma omp parallel 
	{
		int i = 0, j = 0;
		int *window_idxs;
		window_idxs = (int*)malloc(c * c * sizeof(int));

#pragma omp parallel for 
		for (i = 0; i < cols_required; i++) {
			for (j = 0; j < rows_required; j++) {
				Pixel2 local_av;
				double local_r = 0, local_g = 0, local_b = 0;
				unsigned int k = 0, l = 0;

				int col_idx, row_idx, im_idx, relative_index, write_idx;

				col_idx = i * c * im->width; //i*c to skip to next window index and then * width to account for flattened array

				local_av.r = 0; local_av.g = 0; local_av.b = 0;
				row_idx = (j * c);
				im_idx = col_idx + row_idx;

				// iterating through idx of every top right corner of windows
				for (k = 0; k < c; k++) {
					for (l = 0; l < c; l++) {
						relative_index = (k*im->width) + l;
						window_idxs[k*c + l] = im_idx + relative_index;
						local_r = local_r + (double)im->data[im_idx + relative_index].r;
						local_g = local_g + (double)im->data[im_idx + relative_index].g;
						local_b = local_b + (double)im->data[im_idx + relative_index].b;
					}
				}
				local_r = round(local_r / (c_2));
				local_g = round(local_g / (c_2));
				local_b = round(local_b / (c_2));

				local_av.r = (int)local_r;
				local_av.g = (int)local_g;
				local_av.b = (int)local_b;
				local_r = 0, local_g = 0, local_b = 0;

				for (k = 0; k < c*c; k++) { // iterate through the indexes we stored when we were calculating the window average and assign averge values into the image.
					write_idx = window_idxs[k];
					im->data[write_idx] = local_av;
				}

			}
		}
	}
	return(im);
}

static Pixel2 *seq_print_av_col(Image *im) {
	unsigned int i, j;

	//TODO: calculate the average colour value
	double rglob_av = 0, bglob_av = 0, gglob_av = 0, n = (double)im->height * im->width;
	// we access width as second index
	unsigned int idx;
	for (i = 0; i < im->height; i++) {
		for (j = 0; j < im->width; j++) {
			idx = (i*im->width) + j;
			rglob_av = rglob_av + (double)im->data[idx].r;
			gglob_av = gglob_av + (double)im->data[idx].g;
			bglob_av = bglob_av + (double)im->data[idx].b;
		}
	}
	rglob_av = round(rglob_av / n);
	gglob_av = round(gglob_av / n);
	bglob_av = round(bglob_av / n);
	// Output the average colour value for the image
	Pixel2 *glob_av;
	glob_av = (Pixel2*)malloc(sizeof(Pixel2));
	glob_av->r = (int)rglob_av;
	glob_av->g = (int)gglob_av;
	glob_av->b = (int)bglob_av;
	printf("CPU Average image colour red = %d, green = %d, blue = %d \n", glob_av->r, glob_av->g, glob_av->b);
	return glob_av;
}

static Pixel2 *omp_print_av_col(Image *im) {

	double rglob_av = 0, bglob_av = 0, gglob_av = 0, n = (double)im->height * im->width;

	int i, j, idx;

#pragma omp parallel for
	for (i = 0; i < (int)im->height; i++) {
		double tempr = 0, tempg = 0, tempb = 0;
		for (j = 0; j < (int)im->width; j++) {
			idx = (i*im->width) + j;
			if (idx == 0) {
				tempr = (double)im->data[idx].r;
				tempg = (double)im->data[idx].g;
				tempb = (double)im->data[idx].b;
			}
			else {
				tempr = tempr + (double)im->data[idx].r;
				tempg = tempg + (double)im->data[idx].g;
				tempb = tempb + (double)im->data[idx].b;
			}
		}
#pragma omp critical
		{
			rglob_av = rglob_av + tempr;
			gglob_av = gglob_av + tempg;
			bglob_av = bglob_av + tempb;

		}
	}
	rglob_av = rglob_av / n;
	gglob_av = gglob_av / n;
	bglob_av = bglob_av / n;

	Pixel2 *glob_av;
	glob_av = (Pixel2*)malloc(sizeof(Pixel2));
	glob_av->r = (int)round(rglob_av);
	glob_av->g = (int)round(gglob_av);
	glob_av->b = (int)round(bglob_av);
	printf("OMP Average image colour red = %d, green = %d, blue = %d \n", glob_av->r, glob_av->g, glob_av->b);
	return glob_av;
}

void input_image_file(const char* filename, uchar4* image)
{
	FILE *f; //input file handle
	char temp[256];
	unsigned int s;
	int w, h;

	//open the input file and write header info for PPM filetype
	f = fopen(filename, "rb");
	if (f == NULL) {
		fprintf(stderr, "Error opening %s input file\n", filename);
		exit(1);
	}
	fscanf(f, "%s\n", &temp);
	fscanf(f, "%d %d\n", &w, &h);
	fscanf(f, "%d\n", &s);


	for (int x = 0; x < w; x++) {
		for (int y = 0; y < h; y++) {
			int i = x + y * w;
			fread(&image[i], sizeof(unsigned char), 3, f); //only read rgb
														   //image[i].w = 255;
		}
	}

	fclose(f);
}

void output_image_file(uchar4* image, const char* out_name, int w, int h)
{
	FILE *f; //output file handle

			 //open the output file and write header info for PPM filetype
	f = fopen(out_name, "wb");
	if (f == NULL) {
		fprintf(stderr, "Error opening %s output file\n", out_name);
		exit(1);
	}
	fprintf(f, "P6\n");
	fprintf(f, "# COMX521 Assignment 2\n");
	fprintf(f, "%d %d\n%d\n", w, h, 255);
	for (int x = 0; x < w; x++) {
		for (int y = 0; y < h; y++) {
			int i = x + y * w;
			fwrite(&image[i], sizeof(unsigned char), 3, f); //only write rgb (ignoring a)
		}
	}

	fclose(f);
}

void get_file_dims(const char* filename, int* w, int* h)
{
	FILE *f; //input file handle
	char temp[256];
	int imw, imh;

	//open the input file and write header info for PPM filetype
	f = fopen(filename, "rb");
	if (f == NULL) {
		fprintf(stderr, "Error opening %s input file\n", filename);
		exit(1);
	}
	fscanf(f, "%s\n", &temp);
	fscanf(f, "%d %d\n", &imw, &imh);
	*w = imw;
	*h = imh;

	fclose(f);
}

