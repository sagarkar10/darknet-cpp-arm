#include <stdio.h>
extern "C" {
#include "image.hpp"
}
#include "math.h"
#include "limits.h"
#include <string>
#include <iostream>
#include <vector>

#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/core/Types.h"
#include "test_helpers/Utils.h"

//#define STB_IMAGE_IMPLEMENTATION
//#include "stb_image.h"
//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include "stb_image_write.h"

using namespace arm_compute;
using namespace test_helpers;
using namespace std;

enum layer_type {
	CONVOLUTIONAL, MAXPOOL, REGION, SOFTMAX, CONNECTED
};

typedef enum {
	LOGISTIC, RELU, LINEAR, LEAKY_RELU
} ACTIVATION;

typedef struct layer {
	int layer_id = 0;
	layer_type type;
	int batch_normalize = 0;
	int size = 0;
	int stride = 0;
	int pad = 0;
	int filters = 0;
	int batch = 1;
	int in_w = 0, in_h = 0, in_c = 0;
	int out_w = 0, out_h = 0, out_c = 0;
	int padding = size / 2;
	int num = 5;
	float* anchors;
//	anchors = (float*) calloc(num*2,sizeof(float));
	float thresh;
	int classes = 20;
	int coords = 4;
	//layer* prev_layer = NULL;
	ACTIVATION activation = LEAKY_RELU;
} layer;

typedef struct network {
	int src_width = 416;
	int src_height = 416;
	int channels = 3;
	int batch = 1;
	int subdivision = 1;
	int num_layers = 0;
} network;

vector<layer> layers;
network net;
int n_layers = -1;

static inline float linear_activate(float x) {
	return x;
}
static inline float logistic_activate(float x) {
	return 1. / (1. + exp(-x));
}

static inline float relu_activate(float x) {
	return x * (x > 0);
}
static inline float leaky_activate(float x) {
	return (x > 0) ? x : .1 * x;
}

float activate(float x, ACTIVATION a) {
	switch (a) {
	case LINEAR:
		return linear_activate(x);
	case LOGISTIC:
		return logistic_activate(x);
	case RELU:
		return relu_activate(x);
	case LEAKY_RELU:
		return leaky_activate(x);
	default:
		cout << "THIS ACTIVATION IS NOT IMPLEMENTED";
	}
	return 0;
}
void activate_tensor(Tensor& src, ACTIVATION a) {
	Window input_window;
	input_window.use_tensor_dimensions(src.info());
	Iterator input_it(&src, input_window);
	execute_window_loop(input_window,
			[&](const Coordinates & id)
			{
				*reinterpret_cast<float *>(input_it.ptr()) = activate(*reinterpret_cast<float *>(input_it.ptr()), a);
			}, input_it);
}

ACTIVATION string_to_act(string str) {
	if (!str.compare("leaky"))
		return LEAKY_RELU;
	else if (!str.compare("relu"))
		return RELU;
	else if (!str.compare("linear"))
		return LINEAR;
	else {
		cout << "WRONG VALUE....:" << str << endl;
		exit(0);
	}
}

layer_type string_to_layer(string str) {
	if (!str.compare("convolutional"))
		return CONVOLUTIONAL;
	else if (!str.compare("maxpool"))
		return MAXPOOL;
	else if (!str.compare("region"))
		return REGION;
	else {
		cout << "WRONG VALUE....:" << str << endl;
		exit(0);
	}
}

bool isFloat(string myString) {
	std::istringstream iss(myString);
	float f;
	iss >> noskipws >> f; // noskipws considers leading whitespace invalid
	// Check the entire string was consumed and if either failbit or badbit is set
	return iss.eof() && !iss.fail();
}

void get_float_list(int l, string str) {
	unsigned int i = 0, j = 0;
	int c = 0;
	while (c < layers[l].num * 2) {
		i = str.find(",", j);
		layers[l].anchors[c] = (stof(str.substr(j, i)));
		j = i + 1;
		c++;
	}
}
void config_parser(string str) {
	switch (str[0]) {
	case '\n': {
		break;
	}
	case '#': {
		break;
	}
	case '[': {
		string section = str.substr(1, str.size() - 2);
		if ((section.compare("net"))) {
			n_layers++;
			layers.push_back(layer());
			layers[n_layers].type = string_to_layer(section);
			layers[n_layers].layer_id = n_layers;
		}
		break;
	}
	default: {
		string key = str.substr(0, str.find("="));
		string value = str.substr(str.find("=") + 1);
		float int_value = 0;
		try {
			stringstream(value) >> int_value;
		} catch (exception &e) {
			// Exception Handlers
		}
//		cout<<"Setting something:"<<endl;
		if (layers.size() > 0 && layers[n_layers].type != REGION) {
			if (!key.compare("batch_normalize"))
				layers[n_layers].batch_normalize = int_value;
			else if (!key.compare("filters"))
				layers[n_layers].filters = int_value;
			else if (!key.compare("size"))
				layers[n_layers].size = int_value;
			else if (!key.compare("pad"))
				layers[n_layers].pad = int_value;
			else if (!key.compare("stride"))
				layers[n_layers].stride = int_value;
			else if (!key.compare("activation"))
				layers[n_layers].activation = string_to_act(value);

		} else if (layers.size() > 0 && layers[n_layers].type == REGION) {
			if (!key.compare("anchors")) {
				layers[n_layers].anchors = (float*) calloc(
						layers[n_layers].num * 2, sizeof(float));
				get_float_list(n_layers, value);
			} else if (!key.compare("num"))
				layers[n_layers].num = int_value;
			else if (!key.compare("classes"))
				layers[n_layers].classes = int_value;
			else if (!key.compare("thresh"))
				layers[n_layers].thresh = int_value;
			else if (!key.compare("coords"))
				layers[n_layers].coords = int_value;

		} else {
			if (!key.compare("batch"))
				net.batch = int_value;
			else if (!key.compare("subdivision"))
				net.subdivision = int_value;
			else if (!key.compare("width"))
				net.src_width = int_value;
			else if (!key.compare("height"))
				net.src_height = int_value;
			else if (!key.compare("channels"))
				net.channels = int_value;
		}
	}
	}
}

void config_init() {
	for (unsigned int i = 0; i < layers.size(); ++i) {
		if (i == 0) {
			//			cout<<"prev_layer for:"<<i<<"\t NULL"<<endl;
			layers[i].in_c = net.channels;
			layers[i].in_h = net.src_height;
			layers[i].in_w = net.src_width;
			//			cout<<"layer: "<<i<<" Set in_c,h,w:\n"<<layers[i].in_c<<endl<<layers[i].in_h<<endl<<layers[i].in_w<<endl;
		} else {
			//			cout<<"prev_layer for:"<<i<<"\t NOT NULL"<<endl;
			//			cout<<"Prev layer ID: "<<(layers[i].prev_layer)->layer_id<<endl;
			layers[i].in_c = layers[i - 1].out_c;
			layers[i].in_h = layers[i - 1].out_h;
			layers[i].in_w = layers[i - 1].out_w;
			//			cout<<"layer: "<<i<<" Set in_c,h,w:\n"<<layers[i].in_c<<endl<<layers[i].in_h<<endl<<layers[i].in_w<<endl;
		}
		if (layers[i].type == CONVOLUTIONAL) {
			if (layers[i].pad == 1)
				layers[i].padding = layers[i].size / 2;
			else
				layers[i].padding = 0;
			layers[i].out_c = layers[i].filters;
			layers[i].out_h = (layers[i].in_h + 2 * layers[i].padding
					- layers[i].size) / layers[i].stride + 1;
			layers[i].out_w = (layers[i].in_w + 2 * layers[i].padding
					- layers[i].size) / layers[i].stride + 1;
		} else if (layers[i].type == MAXPOOL) {
			if (layers[i].pad == 1)
				layers[i].padding = 1;
			else
				layers[i].padding = (layers[i].size - 1) / 2;
			layers[i].out_c = layers[i].in_c;
			layers[i].out_h = (layers[i].in_h + 2 * layers[i].padding)
					/ (layers[i].stride);
			layers[i].out_w = (layers[i].in_w + 2 * layers[i].padding)
					/ (layers[i].stride);
		} else if (layers[i].type == REGION) {

//			layers[i].in_h = net.src_height;
//			layers[i].in_w = net.src_width;
			layers[i].in_c = layers[i].num
					* (layers[i].classes + layers[i].coords + 1);
			layers[i].out_c = (layers[i].in_c);
			layers[i].out_h = (layers[i].in_h);
			layers[i].out_w = (layers[i].in_w);

		}
	}
}

void dump(Tensor& src, char* filename) {
	int p = 0;
	Window input_window;
	input_window.use_tensor_dimensions(src.info());
	FILE *fp = fopen(filename, "w");
	Iterator input_it(&src, input_window);
	execute_window_loop(input_window, [&](const Coordinates & id)
	{
		p++;
		fprintf(fp,"%f\n",*reinterpret_cast<float *>(input_it.ptr()));
	}, input_it);
	fclose(fp);
	cout << filename << "\tlines: " << p << endl;
}

image load(Tensor& src, char* filename) {
	char buff[256];
	char *input = buff;
	strncpy(input, filename, 256);
	image im = load_image_color(input, 0, 0);
	image sized = letterbox_image(im, 416, 416);
	float *X = sized.data;
	Window input_window;
	input_window.use_tensor_dimensions(src.info());
	Iterator input_it(&src, input_window);
	execute_window_loop(input_window,
			[&](const Coordinates & id)
			{
				*reinterpret_cast<float *>(input_it.ptr()) = X[id.z()*(416*416)+id.y()*416+id.x()];
			}, input_it);
	return sized;
}

void dump_vector(vector<float>&vec, char* filename) {
	cout << filename << endl;
	FILE* fp = fopen(filename, "w");
	for (int i = 0; i < vec.size(); ++i) {
		fprintf(fp, "%f\n", vec[i]);
	}
}

void dump_float(float* data, int size, char* filename) {
	cout << filename << endl;
	FILE* fp = fopen(filename, "w");
	for (int i = 0; i < size; ++i) {
		fprintf(fp, "%f\n", data[i]);
	}
}

void initialize_wb(Tensor& wb, FILE *fp) {
	Window input_window;
	input_window.use_tensor_dimensions(wb.info());
	Iterator input_it(&wb, input_window);
	execute_window_loop(input_window, [&](const Coordinates & id)
	{	float temp = 0;
		fread(&temp, sizeof(float),1,fp);
		*reinterpret_cast<float *>(input_it.ptr()) = temp;
	}, input_it);
}

void initialize_batchnorm_param(vector<float>& a, FILE* fp, int size) {
	float temp = 0;
	for (int i = 0; i < size; i++) {
		fread(&temp, sizeof(float), 1, fp);
		a.push_back(temp);
	}
}
void vector2tensor(Tensor& b, vector<float> &bias) {
	Window input_window;
	input_window.use_tensor_dimensions(b.info());
	int i = 0;
	Iterator input_it(&b, input_window);
	execute_window_loop(input_window, [&](const Coordinates & id)
	{
		(*reinterpret_cast<float *>(input_it.ptr())) = bias[i++];
	}, input_it);
}

void tensor2vector(Tensor& b, vector<float> &bias) {
	bias.clear();
	Window input_window;
	input_window.use_tensor_dimensions(b.info());

	Iterator input_it(&b, input_window);
	execute_window_loop(input_window, [&](const Coordinates & id)
	{
		bias.push_back((*reinterpret_cast<float *>(input_it.ptr())));
	}, input_it);
}
void tensor2array(Tensor& b, float*bias) {
	Window input_window;
	input_window.use_tensor_dimensions(b.info());
	int c = 0;
	Iterator input_it(&b, input_window);
	execute_window_loop(input_window, [&](const Coordinates & id)
	{
		bias[c++] = (*reinterpret_cast<float *>(input_it.ptr()));
	}, input_it);
}

void forward_batchnorm(Tensor& src, vector<float> biases, vector<float> scales,
		vector<float> mean, vector<float> variance, int batch, int n, int h,
		int w, bool onlybias) {
	Window input_window;
	input_window.use_tensor_dimensions(src.info());
	Iterator input_it(&src, input_window);
	for (int z = 0; z < n; ++z) {
		for (int i = 0; i < h; ++i) {
			for (int x = 0; x < w; ++x) {
				if (!onlybias) {
					*reinterpret_cast<float*>(src.buffer()
							+ src.info()->offset_element_in_bytes(
									Coordinates(x, i, z))) =
							(*reinterpret_cast<float*>(src.buffer()
									+ src.info()->offset_element_in_bytes(
											Coordinates(x, i, z))) - mean[z])
									/ (sqrt(variance[z]) + .000001f);
					*reinterpret_cast<float*>(src.buffer()
							+ src.info()->offset_element_in_bytes(
									Coordinates(x, i, z))) *= scales[z];
				}
				*reinterpret_cast<float*>(src.buffer()
						+ src.info()->offset_element_in_bytes(
								Coordinates(x, i, z))) += biases[z];
			}
		}
	}
}

int entry_index(layer l, int batch, int location, int entry) {
	int n = location / (l.in_w * l.in_h);
	int loc = location % (l.in_w * l.in_h);
	return batch * l.out_h * l.out_c * l.out_w
			+ n * l.in_w * l.in_h * (l.coords + l.classes + 1)
			+ entry * l.in_w * l.in_h + loc;
}
void activate_array(vector<float>& x, int index, const int n,
		const ACTIVATION a) {
	int i;
	for (i = index; i < n; ++i) {
		x[i] = activate(x[i], a);
	}
}

void activate_array(float* x, const int n, const ACTIVATION a) {
	int i;
	for (i = 0; i < n; ++i) {
		x[i] = activate(x[i], a);
	}
}

Tensor src;
Tensor weights[10];
Tensor biases[10];
Tensor out_conv[10];
Tensor out_act[10];
Tensor out_pool[6];
Tensor out_fc1;
Tensor out_softmax;
NEConvolutionLayer conv[9];
NEPoolingLayer pool[6];
NEFullyConnectedLayer fc1;
NEActivationLayer act[10];
int filters[9];
int conv_layers[9];
vector<float> rolling_mean[10];
vector<float> rolling_variance[10];
vector<float> scales[10];
vector<float> r_in;
vector<float> biases_vector[10];

void softmax(float *input, int n, float temp, int stride, float *output) {
	int i;
	float sum = 0;
	float largest = -FLT_MAX;
	for (i = 0; i < n; ++i) {
		if (input[i * stride] > largest)
			largest = input[i * stride];
	}
	for (i = 0; i < n; ++i) {
		float e = exp(input[i * stride] / temp - largest / temp);
		sum += e;
		output[i * stride] = e;
	}
	for (i = 0; i < n; ++i) {
		output[i * stride] /= sum;
	}
}

void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups,
		int group_offset, int stride, float temp, float *output) {
	int g, b;
	for (b = 0; b < batch; ++b) {
		for (g = 0; g < groups; ++g) {
			softmax(input + b * batch_offset + g * group_offset, n, temp,
					stride, output + b * batch_offset + g * group_offset);
		}
	}
}

float* forward_region_layer(layer l) {
	int n_out = l.in_c * l.in_h * l.in_w;
	int n_in = n_out;
	int b, n;
	float* output = (float*) calloc(n_out, sizeof(float));
	float* input = (float*) calloc(n_in, sizeof(float));
	tensor2array(out_conv[8], input);
	memcpy(output, input, n_out * l.batch * sizeof(float));
	for (b = 0; b < l.batch; ++b) {
		for (n = 0; n < l.num; ++n) {
			int index = entry_index(l, b, n * l.in_w * l.in_h, 0);
			activate_array(output + index, 2 * l.in_w * l.in_h, LOGISTIC);
			index = entry_index(l, b, n * l.in_w * l.in_h, 4);
			activate_array(output + index, l.in_w * l.in_h, LOGISTIC);
		}
	}
	int index = entry_index(l, 0, 0, 5);
	softmax_cpu(input + index, l.classes, l.batch * l.num, n_in / l.num,
			l.in_w * l.in_h, 1, l.in_w * l.in_h, 1, output + index);
//	dump_float(output,n_out, "dump/output_softmax.txt");
	return output;
}

box get_region_box(float *x, float* biases, int n, int index, int i, int j,
		int w, int h, int stride) {
	box b;
	b.x = (i + x[index + 0 * stride]) / w;
	b.y = (j + x[index + 1 * stride]) / h;
	b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
	b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
	return b;
}

void get_region_boxes(layer l, float* outputs, int w, int h, float thresh,
		float **probs, box *boxes, int only_objectness, int *map,
		float tree_thresh) {
	int i, j, n, z;
	float *predictions = outputs;
	for (i = 0; i < l.in_w * l.in_h; ++i) {
		int row = i / l.in_w;
		int col = i % l.in_w;
		for (n = 0; n < l.num; ++n) {

			int index = n * l.in_w * l.in_h + i;
			int obj_index = entry_index(l, 0, n * l.in_w * l.in_h + i, 4);
			int box_index = entry_index(l, 0, n * l.in_w * l.in_h + i, 0);
			float scale = predictions[obj_index];
			boxes[index] = get_region_box(predictions, l.anchors, n, box_index,
					col, row, l.in_w, l.in_h, l.in_w * l.in_h);
			if (1) {
				int max = w > h ? w : h;
				boxes[index].x = (boxes[index].x - (max - w) / 2. / max)
						/ ((float) w / max);
				boxes[index].y = (boxes[index].y - (max - h) / 2. / max)
						/ ((float) h / max);
				boxes[index].w *= (float) max / w;
				boxes[index].h *= (float) max / h;
			}

			boxes[index].x *= w;
			boxes[index].y *= h;
			boxes[index].w *= w;
			boxes[index].h *= h;

			int class_index = entry_index(l, 0, n * l.in_w * l.in_h + i, 5);
			for (j = 0; j < l.classes; ++j) {
				int class_index = entry_index(l, 0, n * l.in_w * l.in_h + i,
						5 + j);
				float prob = scale * predictions[class_index];
				probs[index][j] = (prob > thresh) ? prob : 0;
			}
			if (only_objectness) {
				probs[index][0] = scale;
			}
		}
	}
}

float sec(clock_t clocks) {
	return (float) clocks / CLOCKS_PER_SEC;
}

image load_weights_biases_bnparams(char* wf = "tiny-yolo-voc.weights",
		char* sf = "data/dog.jpg", bool dump = false) {
	fprintf(stderr, "Loading weights from : %s\n", wf);
	FILE *fp = fopen(wf, "rb");
	if (!fp)
		exit(1);

	int major;
	int minor;
	int revision;
	int seen;
	fread(&major, sizeof(int), 1, fp);
	fread(&minor, sizeof(int), 1, fp);
	fread(&revision, sizeof(int), 1, fp);
	fread(&seen, sizeof(int), 1, fp);

	image im = load(src, sf);
	for (int i = 0; i < 9; ++i) {
		cout << "Loading Weights and Biases For layer: " << i << endl;
		initialize_wb(biases[i], fp);
		tensor2vector(biases[i], biases_vector[i]);
		if (layers[conv_layers[i]].type == CONVOLUTIONAL
				&& layers[conv_layers[i]].size == 1) {
			cout << "Skip BN parm" << endl;
		} else {
			cout << "BN param loading: " << endl;
			initialize_batchnorm_param(scales[i], fp, filters[i]);
			initialize_batchnorm_param(rolling_mean[i], fp, filters[i]);
			initialize_batchnorm_param(rolling_variance[i], fp, filters[i]);
		}
		initialize_wb(weights[i], fp);
//		char dump_name[100];
//		sprintf(dump_name, "dump/biases%d.txt", i);
//		dump(biases[i], dump_name);
//		sprintf(dump_name, "dump/weights%d.txt", i);
//		dump(weights[i], dump_name);
//		sprintf(dump_name, "dump/scales%d.txt", i);
//		dump_vector(scales[i], dump_name);
//		sprintf(dump_name, "dump/r_mean%d.txt", i);
//		dump_vector(rolling_mean[i], dump_name);
//		sprintf(dump_name, "dump/r_variance%d.txt", i);
//		dump_vector(rolling_variance[i], dump_name);

	}
	fclose(fp);
	return im;

}

void allocate_tensors() {
	unsigned int width_src_image = net.src_width;
	unsigned int height_src_image = net.src_height;
	unsigned int ifm_src_img = net.channels;
	const TensorShape src_shape(width_src_image, height_src_image, ifm_src_img);
	src.allocator()->init(TensorInfo(src_shape, ifm_src_img, DataType::F32));
	for (int con = 0; con < 6; ++con) {
		unsigned int kernel_x_conv = layers[2 * con].size;
		unsigned int kernel_y_conv = layers[2 * con].size;
		unsigned int ofm_conv = layers[2 * con].filters;
		filters[con] = layers[2 * con].filters;
		conv_layers[con] = 2 * con;
		const TensorShape weights_shape_conv(kernel_x_conv, kernel_y_conv,
				layers[2 * con].in_c, ofm_conv);
		const TensorShape biases_shape_conv(weights_shape_conv[3]);
		const TensorShape out_shape_conv(layers[2 * con].out_h,
				layers[2 * con].out_w, layers[2 * con].out_c);
		weights[con].allocator()->init(
				TensorInfo(weights_shape_conv, 1, DataType::F32));
		biases[con].allocator()->init(
				TensorInfo(biases_shape_conv, 1, DataType::F32));
		out_conv[con].allocator()->init(
				TensorInfo(out_shape_conv, 1, DataType::F32));
		out_act[con].allocator()->init(
				TensorInfo(out_shape_conv, 1, DataType::F32));
		TensorShape out_shape_pool(layers[2 * con + 1].out_h,
				layers[2 * con + 1].out_w, layers[2 * con + 1].out_c);
		out_pool[con].allocator()->init(
				TensorInfo(out_shape_pool, 1, DataType::F32));
	}
	unsigned int kernel_x_conv7 = layers[12].size;
	unsigned int kernel_y_conv7 = layers[12].size;
	unsigned int ofm_conv7 = layers[12].filters;
	filters[6] = layers[12].filters;
	conv_layers[6] = 12;
	const TensorShape weights_shape_conv7(kernel_x_conv7, kernel_y_conv7,
			layers[12].in_c, ofm_conv7);
	const TensorShape biases_shape_conv7(weights_shape_conv7[3]);
	const TensorShape out_shape_conv7(layers[12].out_h, layers[12].out_w,
			layers[12].out_c);
	weights[6].allocator()->init(
			TensorInfo(weights_shape_conv7, 1, DataType::F32));
	biases[6].allocator()->init(
			TensorInfo(biases_shape_conv7, 1, DataType::F32));
	out_conv[6].allocator()->init(
			TensorInfo(out_shape_conv7, 1, DataType::F32));
	out_act[6].allocator()->init(TensorInfo(out_shape_conv7, 1, DataType::F32));
	unsigned int kernel_x_conv8 = layers[13].size;
	unsigned int kernel_y_conv8 = layers[13].size;
	unsigned int ofm_conv8 = layers[13].filters;
	filters[7] = layers[13].filters;
	conv_layers[7] = 13;
	const TensorShape weights_shape_conv8(kernel_x_conv8, kernel_y_conv8,
			layers[13].in_c, ofm_conv8);
	const TensorShape biases_shape_conv8(weights_shape_conv8[3]);
	const TensorShape out_shape_conv8(layers[13].out_h, layers[13].out_w,
			layers[13].out_c);
	weights[7].allocator()->init(
			TensorInfo(weights_shape_conv8, 1, DataType::F32));
	biases[7].allocator()->init(
			TensorInfo(biases_shape_conv8, 1, DataType::F32));
	out_conv[7].allocator()->init(
			TensorInfo(out_shape_conv8, 1, DataType::F32));
	out_act[7].allocator()->init(TensorInfo(out_shape_conv8, 1, DataType::F32));
	unsigned int kernel_x_conv9 = layers[14].size;
	unsigned int kernel_y_conv9 = layers[14].size;
	unsigned int ofm_conv9 = layers[14].filters;
	filters[8] = layers[14].filters;
	conv_layers[8] = 14;
	const TensorShape weights_shape_conv9(kernel_x_conv9, kernel_y_conv9,
			layers[14].in_c, ofm_conv9);
	const TensorShape biases_shape_conv9(weights_shape_conv9[3]);
	const TensorShape out_shape_conv9(layers[14].out_h, layers[14].out_w,
			layers[14].out_c);
	weights[8].allocator()->init(
			TensorInfo(weights_shape_conv9, 1, DataType::F32));
	biases[8].allocator()->init(
			TensorInfo(biases_shape_conv9, 1, DataType::F32));
	out_conv[8].allocator()->init(
			TensorInfo(out_shape_conv9, 1, DataType::F32));
	out_act[8].allocator()->init(TensorInfo(out_shape_conv9, 1, DataType::F32));
	unsigned int num_labels = 20;
}

void configure_tensors() {
	for (int l = 0; l < 6; ++l) {
		Tensor* temp;
		if (l == 0)
			temp = &src;
		else
			temp = &out_pool[l - 1];

		conv[l].configure(temp, &weights[l], NULL, &out_conv[l],
				PadStrideInfo(layers[2 * l].stride, layers[2 * l].stride,
						layers[2 * l].pad, layers[2 * l].pad));
		//		act[l].configure(&out_conv[l], &out_act[l],
		//			ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU));
		pool[l].configure(&out_conv[l], &out_pool[l],
				PoolingLayerInfo(PoolingType::MAX, layers[2 * l + 1].size,
						PadStrideInfo(layers[2 * l + 1].stride,
								layers[2 * l + 1].stride, layers[2 * l + 1].pad,
								layers[2 * l + 1].pad)));
	}
	conv[6].configure(&out_pool[5], &weights[6], NULL, &out_conv[6],
			PadStrideInfo(layers[12].stride, layers[12].stride, layers[12].pad,
					layers[12].pad));
	//	act[6].configure(&out_conv[6], &out_act[6],
	//				ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU));
	conv[7].configure(&out_conv[6], &weights[7], NULL, &out_conv[7],
			PadStrideInfo(layers[13].stride, layers[13].stride, layers[13].pad,
					layers[13].pad));
	//	act[7].configure(&out_conv[7], &out_act[7],
	//					ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU));
	conv[8].configure(&out_conv[7], &weights[8], NULL, &out_conv[8],
			PadStrideInfo(layers[14].stride, layers[14].stride, layers[14].pad,
					layers[14].pad));
	//	act[8].configure(&out_conv[8], &out_act[8],
	//						ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LINEAR));

}

void allocator_allocate() {
	src.allocator()->allocate();
	for (int i = 0; i < 10; ++i) {
		weights[i].allocator()->allocate();
		biases[i].allocator()->allocate();
		out_conv[i].allocator()->allocate();
		//		out_act[i].allocator()->allocate();
		if (i < 6)
			out_pool[i].allocator()->allocate();
	}
}

void batch_norm_run(int i) {
	if (layers[conv_layers[i]].batch_normalize == 1) {
		//Tensor& src, float* biases, float *scales, float *mean, float *variance, int batch, int n, int size
		forward_batchnorm(out_conv[i], biases_vector[i], scales[i],
				rolling_mean[i], rolling_variance[i],
				layers[conv_layers[i]].batch, layers[conv_layers[i]].filters,
				layers[conv_layers[i]].out_h, layers[conv_layers[i]].out_w,
				false);
		/*
		 * Here upto 8 the BN takes the out of act lastly it works on out_conv
		 * */
	} else if (layers[conv_layers[i]].type == CONVOLUTIONAL
			&& layers[conv_layers[i]].size == 1
			&& layers[conv_layers[i]].batch_normalize != 1) {
		forward_batchnorm(out_conv[i], biases_vector[i], scales[i],
				rolling_mean[i], rolling_variance[i],
				layers[conv_layers[i]].batch, layers[conv_layers[i]].filters,
				layers[conv_layers[i]].out_h, layers[conv_layers[i]].out_w,
				true);
	}
}

void activator_run(int i) {
	if (i != 8)
		activate_tensor(out_conv[i], LEAKY_RELU);
	else
		activate_tensor(out_conv[i], LINEAR);
}

float* forward_pass() {
	for (int i = 0; i < 9; ++i) {
		cout << "Running Conv: " << i << endl;
		conv[i].run();
		cout << "Running Batchnorm: " << i << endl;
		batch_norm_run(i);
		cout << "Running Activation: " << i << endl;
		activator_run(i);
		if (i < 6) {
			cout << "Running Pool: " << i << endl;
			pool[i].run();
		}
	}
	cout << "Running Region Layer" << endl;
	float* output = forward_region_layer(layers[15]);
	return output;
}

char** get_labels() {
	char* names[] = { "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
			"car", "cat", "chair", "cow", "diningtable", "dog", "horse",
			"motorbike", "person", "pottedplant", "sheep", "sofa", "train",
			"tvmonitor" };
	return names;
}

void tiny_yolo_run() {
	allocate_tensors();
	configure_tensors();
	allocator_allocate();
	image im = load_weights_biases_bnparams("tiny-yolo-voc.weights", "dog.jpg",
			false);
	for(int loop=0;loop<20;++loop){
		clock_t time;
		time = clock();
		float* output = forward_pass();
		cout << "Prediction time: %f" << sec(clock() - time) << endl;
	}
	layer l = layers[15];
	box *boxes = (box*) calloc(l.in_w * l.in_h * l.num, sizeof(box));
	float **probs = (float**) calloc(l.in_w * l.in_h * l.num, sizeof(float *));
	float thresh = .35;
	float nms = 0.3;
	printf("test_detector: layers = %d, %d, %d\n", l.in_w, l.in_h, l.num);
	int j=0;
	for (j = 0; j < l.in_w * l.in_h * l.num; ++j)
		probs[j] = (float*) calloc(l.classes + 1, sizeof(float));
	get_region_boxes(l, output, 1, 1, thresh, probs, boxes, 0, 0, thresh);
	if (nms)
		do_nms_sort(boxes, probs, l.in_w * l.in_h * l.num, l.classes, nms);

//	char** names = get_labels();
	char* names[] = { "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
				"car", "cat", "chair", "cow", "diningtable", "dog", "horse",
				"motorbike", "person", "pottedplant", "sheep", "sofa", "train",
				"tvmonitor" };
	image** alphabet = load_alphabet();

	draw_detections(im, l.in_w * l.in_h * l.num, thresh, boxes, probs, names,
			alphabet, l.classes);
	cout<<"predictions saved at prediction.png"<<endl;
	save_image(im, "predictions");
}

void print_network() {
	for (unsigned int i = 0; i < layers.size(); ++i) {
		cout << "Input :" << layers[i].in_h << " x " << layers[i].in_w << " x "
				<< layers[i].in_c << endl;
		cout << "Output :" << layers[i].out_h << " x " << layers[i].out_w
				<< " x " << layers[i].out_c << endl;
		cout << "Filters :" << layers[i].filters << endl;
	}
}

fstream read_config() {
	fstream file("tiny-yolo-voc.cfg");
	int i = 0;
	string str;
	while (getline(file, str)) {
		config_parser(str);
		net.num_layers = layers.size();
		i++;
	}
	return file;
}

int main(int argc, char const *argv[]) {
	fstream file = read_config();
	file.close();
	config_init();
	print_network();
	tiny_yolo_run();
	return 0;
}
