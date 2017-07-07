/*
 * TinyYolo.h
 *
 *  Created on: 06-Jul-2017
 *      Author: sagarkar10
 */
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/core/Types.h"
#include "test_helpers/Utils.h"
#include "image.hpp"
#include "box.hpp"
#ifndef DARKNET_SRC_TINYYOLO_H_
#define DARKNET_SRC_TINYYOLO_H_

using namespace arm_compute;
using namespace test_helpers;
using namespace std;
class TinyYolo {
public:
	TinyYolo(char* cfg_f, char* wt_f);
	TinyYolo(const TinyYolo&);
	typedef enum {
		CONVOLUTIONAL, MAXPOOL, REGION, SOFTMAX, CONNECTED
	} layer_type;

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
		float thresh;
		int classes = 20;
		int coords = 4;
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

	void allocator_allocate();
	void batch_norm_run(int i);
	void tiny_yolo_run(image im);
	image load_image(char* sf);
	void print_network();
	void read_config(char* cfg_f);
	void config_init();
	virtual ~TinyYolo();
private:

	vector<layer> layers;
	network net;
	int n_layers = -1;

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

	static inline float linear_activate(float x);
	static inline float logistic_activate(float x);
	static inline float relu_activate(float x);
	static inline float leaky_activate(float x);
	float activate(float x, ACTIVATION a);
	void activate_tensor(Tensor& src, ACTIVATION a);
	ACTIVATION string_to_act(string str);
	layer_type string_to_layer(string str);
	bool isFloat(string myString);
	void get_float_list(int l, string str);
	void config_parser(string str);
	void dump(Tensor& src, char* filename);
	image load(Tensor& src, char* filename);
	void dump_vector(vector<float>&vec, char* filename);
	void dump_float(float* data, int size, char* filename);
	void initialize_wb(Tensor& wb, FILE *fp);
	void initialize_batchnorm_param(vector<float>& a, FILE* fp, int size);
	void vector2tensor(Tensor& b, vector<float> &bias);
	void tensor2vector(Tensor& b, vector<float> &bias);
	void tensor2array(Tensor& b, float*bias);
	int entry_index(layer l, int batch, int location, int entry);
	void forward_batchnorm(Tensor& src, vector<float> biases,
			vector<float> scales, vector<float> mean, vector<float> variance,
			int batch, int n, int h, int w, bool onlybias);
	void activate_array(float* x, int n, ACTIVATION a);
	void softmax_cpu(float *input, int n, int batch, int batch_offset,
			int groups, int group_offset, int stride, float temp,
			float *output);
	box get_region_box(float *x, float* biases, int n, int index, int i, int j,
			int w, int h, int stride);
	void softmax(float *input, int n, float temp, int stride, float *output);
	float* forward_region_layer(layer l);
	void get_region_boxes(layer l, float* outputs, int w, int h, float thresh,
			float **probs, box *boxes, int only_objectness, int *map,
			float tree_thresh);
	void load_weights_biases_bnparams(char* wf, bool dump);
	void allocate_tensors();
	void configure_tensors();

	void activator_run(int i);
	float* forward_pass();
	char** get_labels();

};

#endif /* DARKNET_SRC_TINYYOLO_H_ */
