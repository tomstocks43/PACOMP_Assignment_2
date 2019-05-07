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


#define FAILURE 0
#define SUCCESS !FAILURE
#define _CRT_SECURE_NO_WARNINGS
#define HEADCHAR 24
#define USER_NAME "acp18ts"		//replace with your user name
#define THREADS 16
#define CUDA_CALL(x) {cudaError_t cuda_error__ = (x); if (cuda_error__) printf("CUDA error: " #x " returned \"%s\"\n", cudaGetErrorString(cuda_error__));}
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

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
static Pixel2 cuda_print_av_col(Image *im);
static Image cuda_mosaic_filter2(Image *im, int mosaic_dim);


inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

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
	//TODO: read input image file (either binary or plain text PPM) 
	Image *im_in;
	im_in = (Image *)malloc(sizeof(Image));
	im_in = import_ppm_hdr(imagein, binmode);

	// checking C is not bigger than image dims
	if (c > im_in->height || c > im_in->width) {
		printf("%d is bigger than image dimensions of %d by %d", c, im_in->height, im_in->width);
		exit(3);
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
	case (OPENMP): {
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
		break;
	}
	case (CUDA): {
		clock_t start, stop;
		//TODO: starting timing here
		start = clock();
		//TODO: calculate the average colour value
		Pixel2 cuda_av = cuda_print_av_col(im_in);
		// Output the average colour value for the image
		Image cuda_im;
		cuda_im = cuda_mosaic_filter2(im_in, c); im_in = &cuda_im;

		//TODO: end timing here
		stop = clock();
		printf("CUDA mode execution time took %.4fs\n", ((double)(stop - start) / CLOCKS_PER_SEC));

		break;
	}
	case (ALL): {
		//TODO: starting timing here

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


		//TODO: starting timing here
		//start = clock();
		//TODO: calculate the average colour value
		//glob_av = omp_print_av_col(im_in);
		// Output the average colour value for the image
		//im_in = pad_array(im_in, c, glob_av);
		//im_in = omp_mosaic_filter(im_in, c);
		//TODO: end timing here
		//stop = clock();
		//printf("OPENMP mode execution time took %.4fs\n", ((double)(stop - start) / CLOCKS_PER_SEC));


		//TODO: starting timing here
		start = clock();
		//TODO: calculate the average colour value
		Pixel2 cuda_av = cuda_print_av_col(im_in);
		// Output the average colour value for the image
		Image cuda_im;
		cuda_im = cuda_mosaic_filter2(im_in, c); im_in = &cuda_im;

		//TODO: end timing here
		stop = clock();
		printf("CUDA mode execution time took %.4fs\n", ((double)(stop - start) / CLOCKS_PER_SEC));




		break;
	}
	}

	//save the output image file (from last executed mode)
	output_ppm(imageout, im_in);
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

__global__ void cuda_av_kern(Image *im, Pixel2 *av, int *numel) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int tempr = 0, tempg = 0, tempb = 0;

	if (idx < *numel) {
		tempr = im->data[idx].r;
		tempg = im->data[idx].g;
		tempb = im->data[idx].b;

		int *r, *g, *b;
		r = &av->r;
		g = &av->g;
		b = &av->b;

		tempr = atomicAdd(r, tempr);
		tempg = atomicAdd(g, tempg);
		tempb = atomicAdd(b, tempb);
		//printf("the value being added is %d %d %d \n", tempr, tempg, tempb);
	}

}

__global__ void cuda_mosaic_kern(Image *im, int *numel) {

	int skip_line = im->width;
	int block_start = (blockIdx.x*blockDim.x) + (blockIdx.y*blockDim.y*skip_line);
	int thread_relative = threadIdx.x + (threadIdx.y*skip_line);
	int idx = block_start + thread_relative;
	//int block_size = blockDim.x*blockDim.y

	int truey = blockIdx.y*blockDim.y + threadIdx.y;
	int truex = blockIdx.x*blockDim.x + threadIdx.x;


	int tempr = 0, tempg = 0, tempb = 0;
	__shared__  int mosaic_r, mosaic_g, mosaic_b, tally;

	/*if (idx == 0) {
		mosaic_r = 0; mosaic_g = 0; mosaic_b = 0;
	}*/
	__syncthreads();

	if (truey < im->height && truex < im->width) {
		tempr = im->data[idx].r;
		tempg = im->data[idx].g;
		tempb = im->data[idx].b;

		int *r, *g, *b, *t;
		r = &mosaic_r;
		g = &mosaic_g;
		b = &mosaic_b;
		t = &tally;

		atomicAdd(r, tempr);
		atomicAdd(g, tempg);
		atomicAdd(b, tempb);
		atomicAdd(t, 1);
	}
	__syncthreads();
	if (truey < im->height && truex < im->width) {
		im->data[idx].r = (int)round((double)mosaic_r / (double)(tally));
		im->data[idx].g = (int)round((double)mosaic_g / (double)(tally));
		im->data[idx].b = (int)round((double)mosaic_b / (double)(tally));

	}
	
}

//__global__ void cuda_mosaic_kern3(Image *im, int *numel) {
//
//	int skip_line = im->width;
//	int block_start = (blockIdx.x*blockDim.x) + (blockIdx.y*blockDim.y*skip_line);
//	int thread_relative = threadIdx.x + (threadIdx.y*skip_line);
//	int idx = block_start + thread_relative;
//	int tid = threadIdx.x + threadIdx.y*blockDim.x;
//
//	int truey = blockIdx.y*blockDim.y + threadIdx.y;
//	int truex = blockIdx.x*blockDim.x + threadIdx.x;
//
//	int tempr = 0, tempg = 0, tempb = 0;
//	extern __shared__  int s_mem[];
//	int size_share = sizeof(s_mem);
//	int*   mosaic_r = (int*)&s_mem[64];
//	__syncthreads();
//
//	if (truey < im->height && truex < im->width) {
//		mosaic_r[tid] = im->data[idx].r;
//		mosaic_g[tid] = im->data[idx].g;
//		mosaic_b[tid] = im->data[idx].b;
//		
//	}
//	__syncthreads();
//	// do reduction in shared mem
//	/*if (truey < im->height && truex < im->width) {
//		for (unsigned int s = 1; s < blockDim.x; s *= 2) {
//			if (tid % (2 * s) == 0) {
//				mosaic_r[tid] += mosaic_r[tid + s];
//				mosaic_g[tid] += mosaic_g[tid + s];
//				mosaic_b[tid] += mosaic_b[tid + s];
//			}
//		}
//	}*/
//	__syncthreads();
//	
//	if (truey < im->height && truex < im->width) {
//		im->data[idx].r = (int)round((double)mosaic_r[0] / (double)(1024));
//		im->data[idx].g = (int)round((double)mosaic_g[0] / (double)(1024));
//		im->data[idx].b = (int)round((double)mosaic_b[0] / (double)(1024));
//	}
//
//}

__global__ void cuda_mosaic_kern4(Image *im, int *numel) {

	int skip_line = im->width;
	int block_start = (blockIdx.x*blockDim.x) + (blockIdx.y*blockDim.y*skip_line);
	int thread_relative = threadIdx.x + (threadIdx.y*skip_line);
	int idx = block_start + thread_relative;

	int truey = blockIdx.y*blockDim.y + threadIdx.y;
	int truex = blockIdx.x*blockDim.x + threadIdx.x;

	int tempr = 0, tempg = 0, tempb = 0;
	extern __shared__  Pixel2 mosaic_av;
	__syncthreads();

	if (truey < im->height && truex < im->width) {
		tempr = im->data[idx].r;
		tempg = im->data[idx].g;
		tempb = im->data[idx].b;

		int *r, *g, *b;
		r = &mosaic_av.r;
		g = &mosaic_av.g;
		b = &mosaic_av.b;

		atomicAdd(r, tempr);
		atomicAdd(g, tempg);
		atomicAdd(b, tempb);
	}
	__syncthreads();
	if (truey < im->height && truex < im->width) {
		im->data[idx].r = (int)round((double)mosaic_av.r / (double)(1024));
		im->data[idx].g = (int)round((double)mosaic_av.g / (double)(1024));
		im->data[idx].b = (int)round((double)mosaic_av.b / (double)(1024));
	}
}

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

static Pixel2 cuda_print_av_col(Image *im) {
	Image h_image;
	Image hd_image;
	Image *d_image;

	h_image = *im;
	hd_image = h_image;

	Pixel2 av;
	av.r = 0; av.g = 0; av.b = 0;
	Pixel2 *d_av;

	int numel = h_image.width*h_image.height;
	int N = (int)round((double)numel / (double)32 );
	int *d_numel;

	//Moving image data to device
	gpuErrchk(cudaMalloc(&hd_image.data, h_image.width*h_image.height * sizeof(Pixel2)));
	gpuErrchk(cudaMemcpy(hd_image.data, h_image.data, h_image.width*h_image.height * sizeof(Pixel2), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc(&d_image, sizeof(Image)));
	gpuErrchk(cudaMemcpy(d_image, &hd_image, sizeof(Image), cudaMemcpyHostToDevice));

	//Moving Pixel for average to device
	gpuErrchk(cudaMalloc(&d_av, sizeof(Pixel2)));
	gpuErrchk(cudaMemcpy(d_av, &av, sizeof(Pixel2), cudaMemcpyHostToDevice));

	gpuErrchk(cudaMalloc(&d_numel, sizeof(int)));
	gpuErrchk(cudaMemcpy(d_numel, &numel, sizeof(int), cudaMemcpyHostToDevice));

	dim3 threadsPerBlock(32, 1, 1); // static blocks of 32 for now

	dim3 blocksPerGrid(N, 1, 1); // uses minimum amount of 32 blocks

	cuda_av_kern << <blocksPerGrid, threadsPerBlock >> > (d_image, d_av, d_numel); // runs kernel
	cudaError_t kern_error = cudaGetLastError();
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemcpy(&av, d_av, sizeof(Pixel2), cudaMemcpyDeviceToHost));

	int avr, avg, avb;
	avr = (int)round((double)av.r / (double)numel);
	avg = (int)round((double)av.g / (double)numel);
	avb = (int)round((double)av.b / (double)numel);

	printf("CUDA Average image colour red = %d, green = %d, blue = %d \n", avr, avg, avb);

	gpuErrchk(cudaFree(hd_image.data));
	hd_image.data = 0;
	gpuErrchk(cudaFree(d_image));
	d_image = 0;

	return av;
}

static Image cuda_mosaic_filter2(Image *im, int mosaic_dim) {
	Image h_image;
	Image hd_image;
	Image *d_image;

	h_image = *im;
	hd_image = h_image;


	int numel = h_image.width*h_image.height;
	int *d_numel;

	

	//Decide how to process based on dimensions
	if (h_image.height * h_image.width <= 1024 && mosaic_dim * mosaic_dim <= 1024) {
		//Moving image data to device
		gpuErrchk(cudaMalloc((void **)&hd_image.data, h_image.width*h_image.height * sizeof(Pixel2)));
		gpuErrchk(cudaMemcpy(hd_image.data, h_image.data, h_image.width*h_image.height * sizeof(Pixel2), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMalloc(&d_image, sizeof(Image)));
		gpuErrchk(cudaMemcpy(d_image, &hd_image, sizeof(Image), cudaMemcpyHostToDevice));

		//moving numel int to device
		gpuErrchk(cudaMalloc(&d_numel, sizeof(int)));
		gpuErrchk(cudaMemcpy(d_numel, &numel, sizeof(int), cudaMemcpyHostToDevice));



		//do 1 thread per mosaic tile
		int numy = (int)ceil((double)im->height / (double)c), numx = (int)ceil((double)im->width / (double)c);
		dim3 threadsPerBlock(numy, numx, 1);
		dim3 blocksPerGrid(1, 1, 1);
		// run kernel
		cuda_simple_mosaic_kern << < blocksPerGrid, threadsPerBlock >> > (d_image, d_numel);
		cudaError_t kern_error = cudaGetLastError();
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		//Bring back modified data from device to host
		gpuErrchk(cudaMemcpy(&h_image, d_image, sizeof(Image), cudaMemcpyDeviceToHost));
		Pixel2* out_image;
		out_image = (Pixel2*)malloc(h_image.height*h_image.width * sizeof(Pixel2));
		gpuErrchk(cudaMemcpy(out_image, d_image->data, h_image.width*h_image.height * sizeof(Pixel2), cudaMemcpyDeviceToHost));

		// free memory
		gpuErrchk(cudaFree(hd_image.data));
		hd_image.data = 0;
		gpuErrchk(cudaFree(d_image));
		d_image = 0;
	}

	else if (h_image.height * h_image.width >= 1024 && mosaic_dim * mosaic_dim <= 1024) {
		//1 block per mosaic tile
		Pixel2 *d_data;
		//calculating how many blocks we need and size of blocks for 1 block = 1 mosaic
		dim3 threadsPerBlock(mosaic_dim, mosaic_dim, 1);
		//int N = (int)round((double)numel / (double)(mosaic_dim*mosaic_dim));
		int numy = (int)ceil((double)im->height / (double)c), numx = (int)ceil((double)im->width / (double)c);
		dim3 blocksPerGrid(numx, numy, 1);

		Pixel2 *mosaic_vals, *d_mosaic_vals;
		mosaic_vals = (Pixel2*)malloc(numy*numx*sizeof(Pixel2));
		gpuErrchk(cudaMalloc((void **)&d_mosaic_vals, numy*numx * sizeof(Pixel2)));
		gpuErrchk(cudaMemset(d_mosaic_vals, 0, numy*numx * sizeof(Pixel2)));

		gpuErrchk(cudaMalloc((void **)&d_data, h_image.width*h_image.height * sizeof(Pixel2)));
		gpuErrchk(cudaMemcpy(d_data, im->data, h_image.width*h_image.height * sizeof(Pixel2), cudaMemcpyHostToDevice));

		gpuErrchk(cudaMalloc(&d_image, sizeof(Image)));
		gpuErrchk(cudaMemcpy(d_image, &h_image, sizeof(Image), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMalloc((void **)&d_mosaic_vals, numy*numx * sizeof(Pixel2)));

		// run kernel and check execution
		cuda_inter_mosaic_kern << <blocksPerGrid, threadsPerBlock >> > (d_image, d_data, d_mosaic_vals); // runs kernel
		cudaError_t kern_error = cudaGetLastError();
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		Pixel2 *h_data;
		h_data = (Pixel2*)malloc(h_image.width*h_image.height * sizeof(Pixel2));

		gpuErrchk(cudaMemcpy(h_data, d_data, h_image.width*h_image.height * sizeof(Pixel2), cudaMemcpyDeviceToHost));
		h_image.data = h_data;

		gpuErrchk(cudaFree(d_data));
		gpuErrchk(cudaFree(d_image));
		

	}



	return h_image;
}





