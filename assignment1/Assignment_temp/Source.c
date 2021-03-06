#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define FAILURE 0
#define SUCCESS !FAILURE
#define _CRT_SECURE_NO_WARNINGS
#define HEADCHAR 24
#define USER_NAME "acp18ts"		//replace with your user name


typedef struct {
	unsigned char r, g, b;
} Pixel;
typedef struct {
	unsigned int height, width, magic, max_val;
	Pixel *data;
} Image;
typedef struct {
	int start, stop;
} Timer;
typedef enum MODE { CPU, OPENMP, CUDA, ALL } MODE;

void print_help();
int process_command_line(int argc, char *argv[]);
static Image *import_ppm_hdr(char *imagein, char *binmode);
int powerof2(unsigned int c);
static Image *pad_array(Image *im, unsigned int c, Pixel *glob_av);
static Image *seq_mosaic_filter(Image *im, unsigned int c);
void output_ppm(char *filename, Image *im);
static Image *omp_mosaic_filter(Image *im, unsigned int c);
static Pixel *seq_print_av_col(Image *im);
static Pixel *omp_print_av_col(Image *im);

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
		int i, j;
		i, j = 0;
		//TODO: starting timing here
		clock_t start, stop;
		start = clock();
		Pixel *glob_av;
		glob_av = (Pixel*)malloc(sizeof(Pixel));
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
		Pixel *glob_av;
		glob_av = (Pixel*)malloc(sizeof(Pixel));
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
		printf("CUDA Implementation not required for assignment part 1\n");
		break;
	}
	case (ALL): {
		int i, j;
		i, j = 0;
		//TODO: starting timing here
		clock_t start, stop;
		start = clock();
		Pixel *glob_av;
		glob_av = (Pixel*)malloc(sizeof(Pixel));
		glob_av = seq_print_av_col(im_in);
		//Start the mosaic filter
		im_in = pad_array(im_in, c, glob_av);
		im_in = seq_mosaic_filter(im_in, c);
		//TODO: end timing here
		stop = clock();
		printf("CPU mode execution time took %.4fs\n", ((double)(stop - start) / CLOCKS_PER_SEC));

		//TODO: starting timing here
		start = clock();
		//TODO: calculate the average colour value
		glob_av = omp_print_av_col(im_in);
		// Output the average colour value for the image
		im_in = pad_array(im_in, c, glob_av);
		im_in = omp_mosaic_filter(im_in, c);
		//TODO: end timing here
		stop = clock();
		printf("OPENMP mode execution time took %.4fs\n", ((double)(stop - start) / CLOCKS_PER_SEC));
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
	binmode = "PPM_BINARY";
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

	if (strcmp(binmode, "PPM_Binary") == 0) {
		file_point = fopen(imagein, "rb");
		if (file_point == NULL) {
			printf("Could not open file %s", imagein);
			exit(4);
		}
	}
	else if (strcmp(binmode, "PPM_Plain_Text") == 0) {
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
	sscanf(temp, "%d", &max_val); // assign next found number to max

								  //while (fgetc(file_point) != '\n');

	im->data = (Pixel*)malloc(width * height * sizeof(Pixel)); //memory needed for image = number of elements * memory size of 1 pixel

	if (!im) { // check if mem allocation fails
		printf("test");
		printf("Unable to allocate memory for image data \n");
		exit(7);
	}

	//read pixel data from file
	if (strcmp(binmode, "PPM_Binary") == 0) {
		if (fread(im->data, 3 * height, width, file_point) != width) { // reads into img data, the number of elements to be read is 3 (rgb) * number of x elements because each element in the array is a y row. Thus third arg is the size of a y row.
			fprintf(stderr, "Error loading image '%s'\n", imagein);
			exit(8);
		}
	}


	else if (strcmp(binmode, "PPM_Plain_Text") == 0) {
		for (i = 0; i < width * height; i++) {
			 fscanf(file_point, "%d %d %d", &im->data[i].r, &im->data[i].g, &im->data[i].b);
		}
	}
	im->magic = magic[1] - '0';
	im->height = height;
	im->width = width;
	im->max_val = max_val;

	fclose(file_point);
	return im;
}

int powerof2(unsigned int c)
{
	while (((c % 2) == 0) && c > 1)
		c /= 2;
	return (c == 1);
}

static Image *pad_array(Image *im, unsigned int c, Pixel *glob_av) {

	unsigned int rows_required, cols_required;
	rows_required = (unsigned int)ceil((double)im->height / c);
	cols_required = (unsigned int)ceil((double)im->width / c);
	unsigned int windows_width = cols_required * c;
	unsigned int windows_height = rows_required * c;
	int pad_sides = (int)ceil((float)(windows_width - im->width) / 2);
	int pad_height = (int)ceil((float)(windows_height - im->height) / 2);

	if (pad_height != 0 || pad_sides != 0) {
		Pixel* padded;
		padded = (Pixel*)malloc(windows_width * windows_height * sizeof(Pixel));
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
	Pixel *local_av;
	local_av = (Pixel*)malloc(sizeof(Pixel));


	for (i = 0; i < cols_required; i++) {
		col_idx = i * c * im->width; //i*c to skip to next window index and then * width to account for flattened array
		for (j = 0; j < rows_required; j++) {
			local_av->r = 0; local_av->g = 0; local_av->b = 0;

			row_idx = (j * c);
			im_idx = col_idx + row_idx;
			// iterating through idx of every top right corner of windows
			for (k = 0; k < c; k++) {
				for (l = 0; l < c; l++) {
					relative_index = (k*im->width) + l;
					window_idxs[k*c + l] = im_idx + relative_index;
					local_av->r = local_av->r + im->data[im_idx + (k * im->width) + l].r / (c*c);
					local_av->g = local_av->g + im->data[im_idx + (k * im->width) + l].g / (c*c);
					local_av->b = local_av->b + im->data[im_idx + (k * im->width) + l].b / (c*c);
				}
			}
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

	// pixel data
	/*unsigned int i;
	for (i = 0; i < im->width * im->height; i++) {
		fprintf(file_point, "%d %d %d ", im->data->r, im->data->g, im->data->b);
	}*/
	fwrite(im->data, 3 * im->width, im->height, file_point);
	fclose(file_point);
}

static Image *omp_mosaic_filter(Image *im, unsigned int c) {
	// need to find indices of where mosiac windows will be applied so we can iterate
	unsigned int rows_required, cols_required;
	rows_required = im->height / c;
	cols_required = im->width / c;
	int *height_idxs, *width_idxs;
	height_idxs = (int*)malloc(rows_required * sizeof(int));
	width_idxs = (int*)malloc(cols_required * sizeof(int));
	unsigned int n;
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
				Pixel local_av;
				int k = 0, l = 0;

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
						local_av.r = local_av.r + im->data[im_idx + relative_index].r / (c*c);
						local_av.g = local_av.g + im->data[im_idx + relative_index].g / (c*c);
						local_av.b = local_av.b + im->data[im_idx + relative_index].b / (c*c);
					}
				}
				for (k = 0; k < c*c; k++) { // iterate through the indexes we stored when we were calculating the window average and assign averge values into the image.
					write_idx = window_idxs[k];
					im->data[write_idx] = local_av;
				}

			}
		}
	}
	return(im);
}

static Pixel *seq_print_av_col(Image *im) {
	unsigned int i, j;
	i, j = 0;
	//TODO: calculate the average colour value
	int rglob_av = 0, bglob_av = 0, gglob_av = 0;
	// we access width as second index
	unsigned int n, idx;
	n = im->width * im->height;
	for (i = 0; i < im->height; i++) {
		for (j = 0; j < im->width; j++) {
			idx = (i*im->width) + j;
			rglob_av = rglob_av + (unsigned int)im->data[idx].r;
			gglob_av = gglob_av + (unsigned int)im->data[idx].g;
			bglob_av = bglob_av + (unsigned int)im->data[idx].b;
		}
	}
	rglob_av = rglob_av / (im->height * im->width);
	gglob_av = gglob_av / (im->height * im->width);
	bglob_av = bglob_av / (im->height * im->width);
	// Output the average colour value for the image
	printf("CPU Average image colour red = %d, green = %d, blue = %d \n", rglob_av, gglob_av, bglob_av);
	Pixel *glob_av;
	glob_av = (Pixel*)malloc(sizeof(Pixel));
	glob_av->r = rglob_av;
	glob_av->g = gglob_av;
	glob_av->b = bglob_av;
	return glob_av;
}

static Pixel *omp_print_av_col(Image *im) {

	unsigned int rglob_av = 0, bglob_av = 0, gglob_av = 0;
	int i, j, n, idx;
	n = im->width * im->height;
#pragma omp parallel for 
	for (i = 0; i < im->height; i++) {
		int tempr = 0, tempg = 0, tempb = 0;
		for (j = 0; j < im->width; j++) {
			idx = (i*im->width) + j;
			tempr = tempr + (unsigned int)im->data[idx].r;
			tempg = tempg + (unsigned int)im->data[idx].g;
			tempb = tempb + (unsigned int)im->data[idx].b;
		}
#pragma omp critical
		{
			rglob_av = rglob_av + tempr;
			gglob_av = gglob_av + tempg;
			bglob_av = bglob_av + tempb;
		}
	}
	rglob_av = rglob_av / (im->height * im->width);
	gglob_av = gglob_av / (im->height * im->width);
	bglob_av = bglob_av / (im->height * im->width);
	printf("OMP Average image colour red = %d, green = %d, blue = %d \n", rglob_av, gglob_av, bglob_av);
	Pixel *glob_av;
	glob_av = (Pixel*)malloc(sizeof(Pixel));
	glob_av->r = rglob_av;
	glob_av->g = gglob_av;
	glob_av->b = bglob_av;
	return glob_av;
}