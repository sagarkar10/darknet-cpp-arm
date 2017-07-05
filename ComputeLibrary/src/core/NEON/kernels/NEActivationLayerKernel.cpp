/*
 * Copyright (c) 2017 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/core/NEON/kernels/NEActivationLayerKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEMath.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include <iostream>
#include <arm_neon.h>
#include <array>
#include <map>

using namespace arm_compute;


void print_float32x4 (float32x4_t data) {
    int i;
    static float p[4];

    vst1q_f32 (p, data);

    //printf ("%s = ", name);
    for (i = 0; i < 4; i++) {
        printf ("%f ", p[i]);
    }
    printf ("\n");
}



NEActivationLayerKernel::NEActivationLayerKernel()
    : _func(nullptr), _act_info(ActivationFunction::LOGISTIC)
{
}

void NEActivationLayerKernel::configure(const ITensor *input, ITensor *output, ActivationLayerInfo activation_info)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

    static std::map<ActivationFunction, ActivationFunctionExecutorPtr> act_map =
    {
        { ActivationFunction::ABS, &NEActivationLayerKernel::activation<ActivationFunction::ABS> },
        { ActivationFunction::LINEAR, &NEActivationLayerKernel::activation<ActivationFunction::LINEAR> },
        { ActivationFunction::LOGISTIC, &NEActivationLayerKernel::activation<ActivationFunction::LOGISTIC> },
        { ActivationFunction::RELU, &NEActivationLayerKernel::activation<ActivationFunction::RELU> },
        { ActivationFunction::BOUNDED_RELU, &NEActivationLayerKernel::activation<ActivationFunction::BOUNDED_RELU> },
        { ActivationFunction::LEAKY_RELU, &NEActivationLayerKernel::activation<ActivationFunction::LEAKY_RELU> },
		{ ActivationFunction::NOTHING, &NEActivationLayerKernel::activation<ActivationFunction::NOTHING> },
		{ ActivationFunction::SOFT_RELU, &NEActivationLayerKernel::activation<ActivationFunction::SOFT_RELU> },
        { ActivationFunction::SQRT, &NEActivationLayerKernel::activation<ActivationFunction::SQRT> },
        { ActivationFunction::SQUARE, &NEActivationLayerKernel::activation<ActivationFunction::SQUARE> },
        { ActivationFunction::TANH, &NEActivationLayerKernel::activation<ActivationFunction::TANH> },
    };
    _input    = input;
    _output   = output;
    _func     = act_map[activation_info.activation()];
    _act_info = activation_info;

    constexpr unsigned int num_elems_processed_per_iteration = 16;

    INESimpleKernel::configure(_input, _output, num_elems_processed_per_iteration);
}

template <ActivationLayerInfo::ActivationFunction F>
void NEActivationLayerKernel::activation(const Window &window)
{
    Iterator input(_input, window);
    Iterator output(_output, window);

    static const float32x4_t CONST_1 = vdupq_n_f32(1.f); // 1.f
    static const float32x4_t CONST_01 = vdupq_n_f32(0.1f); // 0.1f
    static const float32x4_t CONST_0 = vdupq_n_f32(0.f);
    const float32x4_t        a       = vdupq_n_f32(_act_info.a());
    const float32x4_t        b       = vdupq_n_f32(_act_info.b());

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const auto input_ptr  = reinterpret_cast<const float *>(input.ptr());
        const auto output_ptr = reinterpret_cast<float *>(output.ptr());

        const float32x4x4_t in  = vld4q_f32(input_ptr);
        float32x4x4_t       tmp = { {} };

        switch(F)
        {
            case ActivationFunction::ABS:
                tmp =
                {
                    {
                        vabsq_f32(in.val[0]),
                        vabsq_f32(in.val[1]),
                        vabsq_f32(in.val[2]),
                        vabsq_f32(in.val[3]),
                    }
                };
                break;
            case ActivationFunction::BOUNDED_RELU:
                tmp =
                {
                    {
                        vminq_f32(a, vmaxq_f32(CONST_0, in.val[0])),
                        vminq_f32(a, vmaxq_f32(CONST_0, in.val[1])),
                        vminq_f32(a, vmaxq_f32(CONST_0, in.val[2])),
                        vminq_f32(a, vmaxq_f32(CONST_0, in.val[3])),
                    }
                };
                break;
            case ActivationFunction::LINEAR:
                tmp =
                {
                    {
                        vmlaq_f32(b, a, in.val[0]),
                        vmlaq_f32(b, a, in.val[1]),
                        vmlaq_f32(b, a, in.val[2]),
                        vmlaq_f32(b, a, in.val[3]),
                    }
                };
                break;
            case ActivationFunction::LOGISTIC:
                tmp =
                {
                    {
                        vinvq_f32(vaddq_f32(CONST_1, vexpq_f32(vnegq_f32(in.val[0])))),
                        vinvq_f32(vaddq_f32(CONST_1, vexpq_f32(vnegq_f32(in.val[1])))),
                        vinvq_f32(vaddq_f32(CONST_1, vexpq_f32(vnegq_f32(in.val[2])))),
                        vinvq_f32(vaddq_f32(CONST_1, vexpq_f32(vnegq_f32(in.val[3])))),
                    }
                };
                break;
            case ActivationFunction::LEAKY_RELU:
                tmp =
                {
                    {
                        vmaxq_f32(vmulq_f32(in.val[0], CONST_01), in.val[0]),
                        vmaxq_f32(vmulq_f32(in.val[1], CONST_01), in.val[1]),
                        vmaxq_f32(vmulq_f32(in.val[2], CONST_01), in.val[2]),
                        vmaxq_f32(vmulq_f32(in.val[3], CONST_01), in.val[3]),
                    }
                };
                print_float32x4(vmaxq_f32(vmulq_f32(in.val[0], CONST_01), in.val[0]));
                break;
            case ActivationFunction::NOTHING:
				tmp =
				{
					{
					vmulq_f32(CONST_1, in.val[0]),
					vmulq_f32(CONST_1, in.val[1]),
					vmulq_f32(CONST_1, in.val[2]),
					vmulq_f32(CONST_1, in.val[3]),
					}
				};
				break;
            case ActivationFunction::RELU:
                tmp =
                {
                    {
                        vmaxq_f32(CONST_0, in.val[0]),
                        vmaxq_f32(CONST_0, in.val[1]),
                        vmaxq_f32(CONST_0, in.val[2]),
                        vmaxq_f32(CONST_0, in.val[3]),
                    }
                };
                break;
            case ActivationFunction::SOFT_RELU:
                tmp =
                {
                    {
                        vlogq_f32(vaddq_f32(CONST_1, vexpq_f32(in.val[0]))),
                        vlogq_f32(vaddq_f32(CONST_1, vexpq_f32(in.val[1]))),
                        vlogq_f32(vaddq_f32(CONST_1, vexpq_f32(in.val[2]))),
                        vlogq_f32(vaddq_f32(CONST_1, vexpq_f32(in.val[3]))),
                    }
                };
                break;
            case ActivationFunction::SQRT:
                tmp =
                {
                    {
                        vinvq_f32(vinvsqrtq_f32(in.val[0])),
                        vinvq_f32(vinvsqrtq_f32(in.val[1])),
                        vinvq_f32(vinvsqrtq_f32(in.val[2])),
                        vinvq_f32(vinvsqrtq_f32(in.val[3])),
                    }
                };
                break;
            case ActivationFunction::SQUARE:
                tmp =
                {
                    {
                        vmulq_f32(in.val[0], in.val[0]),
                        vmulq_f32(in.val[1], in.val[1]),
                        vmulq_f32(in.val[2], in.val[2]),
                        vmulq_f32(in.val[3], in.val[3]),
                    }
                };
                break;
            case ActivationFunction::TANH:
                tmp =
                {
                    {
                        vmulq_f32(a, vtanhq_f32(vmulq_f32(b, in.val[0]))),
                        vmulq_f32(a, vtanhq_f32(vmulq_f32(b, in.val[1]))),
                        vmulq_f32(a, vtanhq_f32(vmulq_f32(b, in.val[2]))),
                        vmulq_f32(a, vtanhq_f32(vmulq_f32(b, in.val[3]))),
                    }
                };
                break;
            default:
                break;
        }

        vst4q_f32(output_ptr, tmp);
    },
    input, output);
}

void NEActivationLayerKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INESimpleKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (this->*_func)(window);
}
