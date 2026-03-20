/**
  ******************************************************************************
  * @file    plants_village_pt.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-03-20T15:16:33+0200
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "plants_village_pt.h"
#include "plants_village_pt_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_plants_village_pt
 
#undef AI_PLANTS_VILLAGE_PT_MODEL_SIGNATURE
#define AI_PLANTS_VILLAGE_PT_MODEL_SIGNATURE     "0xdff34265d510d2ebd55f999196a2ed0e"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2026-03-20T15:16:33+0200"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_PLANTS_VILLAGE_PT_N_BATCHES
#define AI_PLANTS_VILLAGE_PT_N_BATCHES         (1)

static ai_ptr g_plants_village_pt_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_plants_village_pt_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_input0_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 768, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  in_pad_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 768, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  pad_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 972, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  transpose_1_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 973, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  transpose_4_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  nl_5_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  in_pad_6_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  pad_6_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1600, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  transpose_7_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1600, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  transpose_10_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  nl_11_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  gemm_13_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  gemm_14_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 15, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 432, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 16, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4608, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  gemm_13_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16384, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  gemm_13_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  gemm_14_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 480, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  gemm_14_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 15, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1196, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6144, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  gemm_13_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 512, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  gemm_14_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 32, AI_STATIC)

/**  Array metadata declarations section  *************************************/
/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_2_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0476752370595932f),
    AI_PACK_INTQ_ZP(91)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_2_scratch1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0476752370595932f),
    AI_PACK_INTQ_ZP(91)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_2_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 16,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0057524219155311584f, 0.0042476411908864975f, 0.003987650386989117f, 0.0042879777029156685f, 0.00366059597581625f, 0.006142558995634317f, 0.005796386860311031f, 0.0050958977080881596f, 0.0038652336224913597f, 0.004414097405970097f, 0.005132691003382206f, 0.005564449820667505f, 0.0037318202666938305f, 0.00531379459425807f, 0.004631811287254095f, 0.004953495226800442f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_8_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.07832098007202148f),
    AI_PACK_INTQ_ZP(88)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_8_scratch1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.07832098007202148f),
    AI_PACK_INTQ_ZP(88)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_8_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 32,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01280822791159153f, 0.008999011479318142f, 0.007859579287469387f, 0.012034191749989986f, 0.015584812499582767f, 0.010196486487984657f, 0.008526874706149101f, 0.012571031227707863f, 0.008329973556101322f, 0.008121449500322342f, 0.01347044575959444f, 0.014648176729679108f, 0.008416103199124336f, 0.008003013208508492f, 0.01590416021645069f, 0.008725338615477085f, 0.008972587995231152f, 0.016243670135736465f, 0.00841572042554617f, 0.015745854005217552f, 0.009726865217089653f, 0.01033888291567564f, 0.007669347338378429f, 0.011991875246167183f, 0.008525998331606388f, 0.009237975813448429f, 0.012399369850754738f, 0.014291523955762386f, 0.009628417901694775f, 0.010811029933393002f, 0.012957321479916573f, 0.008253918029367924f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_13_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1488724797964096f),
    AI_PACK_INTQ_ZP(29)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_13_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.017427505925297737f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_14_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.5393611788749695f),
    AI_PACK_INTQ_ZP(57)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_14_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.017463473603129387f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(in_pad_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003613994689658284f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(in_pad_6_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007025895640254021f),
    AI_PACK_INTQ_ZP(-114)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_11_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.012692820280790329f),
    AI_PACK_INTQ_ZP(-117)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_5_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007025895640254021f),
    AI_PACK_INTQ_ZP(-114)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pad_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003613994689658284f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pad_6_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007025895640254021f),
    AI_PACK_INTQ_ZP(-114)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(serving_default_input0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003613994689658284f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #17 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_10_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.07832098007202148f),
    AI_PACK_INTQ_ZP(88)))

/* Int quant #18 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_1_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003613994689658284f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #19 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_4_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0476752370595932f),
    AI_PACK_INTQ_ZP(91)))

/* Int quant #20 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(transpose_7_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007025895640254021f),
    AI_PACK_INTQ_ZP(-114)))

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_bias, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2d_2_bias_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_output, AI_STATIC,
  1, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 8, 8), AI_STRIDE_INIT(4, 1, 1, 16, 128),
  1, &conv2d_2_output_array, &conv2d_2_output_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_scratch0, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 1196, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1196, 1196),
  1, &conv2d_2_scratch0_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_scratch1, AI_STATIC,
  3, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 16, 2), AI_STRIDE_INIT(4, 1, 1, 16, 256),
  1, &conv2d_2_scratch1_array, &conv2d_2_scratch1_array_intq)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_weights, AI_STATIC,
  4, 0x1,
  AI_SHAPE_INIT(4, 3, 3, 3, 16), AI_STRIDE_INIT(4, 1, 3, 48, 144),
  1, &conv2d_2_weights_array, &conv2d_2_weights_array_intq)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_bias, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_8_bias_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_output, AI_STATIC,
  6, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 4, 4), AI_STRIDE_INIT(4, 1, 1, 32, 128),
  1, &conv2d_8_output_array, &conv2d_8_output_array_intq)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_scratch0, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 6144, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6144, 6144),
  1, &conv2d_8_scratch0_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_scratch1, AI_STATIC,
  8, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 8, 2), AI_STRIDE_INIT(4, 1, 1, 32, 256),
  1, &conv2d_8_scratch1_array, &conv2d_8_scratch1_array_intq)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_weights, AI_STATIC,
  9, 0x1,
  AI_SHAPE_INIT(4, 16, 3, 3, 32), AI_STRIDE_INIT(4, 1, 16, 512, 1536),
  1, &conv2d_8_weights_array, &conv2d_8_weights_array_intq)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  gemm_13_bias, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &gemm_13_bias_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  gemm_13_output, AI_STATIC,
  11, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &gemm_13_output_array, &gemm_13_output_array_intq)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  gemm_13_scratch0, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 512, 1, 1), AI_STRIDE_INIT(4, 2, 2, 1024, 1024),
  1, &gemm_13_scratch0_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  gemm_13_weights, AI_STATIC,
  13, 0x1,
  AI_SHAPE_INIT(4, 512, 32, 1, 1), AI_STRIDE_INIT(4, 1, 512, 16384, 16384),
  1, &gemm_13_weights_array, &gemm_13_weights_array_intq)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  gemm_14_bias, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 15, 1, 1), AI_STRIDE_INIT(4, 4, 4, 60, 60),
  1, &gemm_14_bias_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  gemm_14_output, AI_STATIC,
  15, 0x1,
  AI_SHAPE_INIT(4, 1, 15, 1, 1), AI_STRIDE_INIT(4, 1, 1, 15, 15),
  1, &gemm_14_output_array, &gemm_14_output_array_intq)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  gemm_14_scratch0, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 2, 2, 64, 64),
  1, &gemm_14_scratch0_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  gemm_14_weights, AI_STATIC,
  17, 0x1,
  AI_SHAPE_INIT(4, 32, 15, 1, 1), AI_STRIDE_INIT(4, 1, 32, 480, 480),
  1, &gemm_14_weights_array, &gemm_14_weights_array_intq)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  in_pad_0_output, AI_STATIC,
  18, 0x1,
  AI_SHAPE_INIT(4, 1, 3, 16, 16), AI_STRIDE_INIT(4, 1, 1, 3, 48),
  1, &in_pad_0_output_array, &in_pad_0_output_array_intq)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  in_pad_6_output, AI_STATIC,
  19, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 8, 8), AI_STRIDE_INIT(4, 1, 1, 16, 128),
  1, &in_pad_6_output_array, &in_pad_6_output_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  nl_11_output, AI_STATIC,
  20, 0x1,
  AI_SHAPE_INIT(4, 1, 4, 4, 32), AI_STRIDE_INIT(4, 1, 1, 4, 16),
  1, &nl_11_output_array, &nl_11_output_array_intq)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  nl_11_output0, AI_STATIC,
  21, 0x1,
  AI_SHAPE_INIT(4, 1, 512, 1, 1), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &nl_11_output_array, &nl_11_output_array_intq)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  nl_5_output, AI_STATIC,
  22, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 8, 16), AI_STRIDE_INIT(4, 1, 1, 8, 64),
  1, &nl_5_output_array, &nl_5_output_array_intq)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  pad_0_output, AI_STATIC,
  23, 0x1,
  AI_SHAPE_INIT(4, 1, 3, 18, 18), AI_STRIDE_INIT(4, 1, 1, 3, 54),
  1, &pad_0_output_array, &pad_0_output_array_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  pad_6_output, AI_STATIC,
  24, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 10, 10), AI_STRIDE_INIT(4, 1, 1, 16, 160),
  1, &pad_6_output_array, &pad_6_output_array_intq)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_input0_output, AI_STATIC,
  25, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 16, 3), AI_STRIDE_INIT(4, 1, 1, 16, 256),
  1, &serving_default_input0_output_array, &serving_default_input0_output_array_intq)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  transpose_10_output, AI_STATIC,
  26, 0x1,
  AI_SHAPE_INIT(4, 1, 4, 4, 32), AI_STRIDE_INIT(4, 1, 1, 4, 16),
  1, &transpose_10_output_array, &transpose_10_output_array_intq)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  transpose_1_output, AI_STATIC,
  27, 0x1,
  AI_SHAPE_INIT(4, 1, 3, 18, 18), AI_STRIDE_INIT(4, 1, 1, 3, 54),
  1, &transpose_1_output_array, &transpose_1_output_array_intq)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  transpose_4_output, AI_STATIC,
  28, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 8, 16), AI_STRIDE_INIT(4, 1, 1, 8, 64),
  1, &transpose_4_output_array, &transpose_4_output_array_intq)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  transpose_7_output, AI_STATIC,
  29, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 10, 10), AI_STRIDE_INIT(4, 1, 1, 16, 160),
  1, &transpose_7_output_array, &transpose_7_output_array_intq)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_14_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_13_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_14_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_14_weights, &gemm_14_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_14_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_14_layer, 14,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA,
  &gemm_14_chain,
  NULL, &gemm_14_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_13_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_11_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_13_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_13_weights, &gemm_13_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_13_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_13_layer, 13,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA,
  &gemm_13_chain,
  NULL, &gemm_14_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_11_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -126, -126, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -124, -124, -124, -124, -124, -124, -124, -124, -124, -124, -124, -124, -124, -124, -124, -124, -123, -123, -123, -123, -123, -123, -123, -123, -123, -123, -123, -123, -123, -123, -123, -123, -122, -122, -122, -122, -122, -122, -122, -122, -122, -122, -122, -122, -122, -122, -122, -122, -122, -121, -121, -121, -121, -121, -121, -121, -121, -121, -121, -121, -121, -121, -121, -121, -121, -120, -120, -120, -120, -120, -120, -120, -120, -120, -120, -120, -120, -120, -120, -120, -120, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -118, -118, -118, -118, -118, -118, -118, -118, -118, -118, -118, -118, -118, -118, -118, -118, -117, -117, -117, -117, -117, -117, -117, -117, -117, -111, -105, -98, -92, -86, -80, -74, -68, -61, -55, -49, -43, -37, -31, -24, -18, -12, -6, 0, 6, 13, 19, 25, 31, 37, 43, 50, 56, 62, 68, 74, 80, 87, 93, 99, 105, 111, 117, 124 };
AI_ARRAY_OBJ_DECLARE(
    nl_11_nl_params, AI_ARRAY_FORMAT_S8,
    nl_11_nl_params_data, nl_11_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_11_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_10_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_11_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_11_layer, 11,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_11_chain,
  NULL, &gemm_13_layer, AI_STATIC, 
  .nl_params = &nl_11_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_10_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_8_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_10_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_10_layer, 10,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_10_chain,
  NULL, &nl_11_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_8_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_8_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_8_weights, &conv2d_8_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_8_scratch0, &conv2d_8_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_8_layer, 9,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool,  forward_conv2d_deep_3x3_sssa8_ch_nl_pool,
  &conv2d_8_chain,
  NULL, &transpose_10_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_integer_INT8), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_7_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pad_6_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_7_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_7_layer, 7,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_7_chain,
  NULL, &conv2d_8_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)


AI_STATIC_CONST ai_i8 pad_6_value_data[] = { -114 };
AI_ARRAY_OBJ_DECLARE(
    pad_6_value, AI_ARRAY_FORMAT_S8,
    pad_6_value_data, pad_6_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  pad_6_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &in_pad_6_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pad_6_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pad_6_layer, 6,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &pad_6_chain,
  NULL, &transpose_7_layer, AI_STATIC, 
  .value = &pad_6_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  in_pad_6_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &in_pad_6_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  in_pad_6_layer, 6,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &in_pad_6_chain,
  NULL, &pad_6_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)


AI_STATIC_CONST ai_i8 nl_5_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -126, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -125, -124, -124, -124, -124, -124, -124, -124, -124, -124, -124, -124, -124, -124, -124, -123, -123, -123, -123, -123, -123, -123, -123, -123, -123, -123, -123, -123, -123, -123, -122, -122, -122, -122, -122, -122, -122, -122, -122, -122, -122, -122, -122, -122, -122, -121, -121, -121, -121, -121, -121, -121, -121, -121, -121, -121, -121, -121, -121, -121, -120, -120, -120, -120, -120, -120, -120, -120, -120, -120, -120, -120, -120, -120, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -119, -118, -118, -118, -118, -118, -118, -118, -118, -118, -118, -118, -118, -118, -118, -118, -117, -117, -117, -117, -117, -117, -117, -117, -117, -117, -117, -117, -117, -117, -117, -116, -116, -116, -116, -116, -116, -116, -116, -116, -116, -116, -116, -116, -116, -115, -115, -115, -115, -115, -115, -115, -115, -115, -115, -115, -115, -115, -115, -115, -114, -114, -114, -114, -114, -114, -114, -114, -107, -100, -94, -87, -80, -73, -67, -60, -53, -46, -39, -33, -26, -19, -12, -5, 1, 8, 15, 22, 28, 35, 42, 49, 56, 62, 69, 76, 83, 90, 96, 103, 110, 117, 123, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_5_nl_params, AI_ARRAY_FORMAT_S8,
    nl_5_nl_params_data, nl_5_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_5_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_5_layer, 5,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_5_chain,
  NULL, &in_pad_6_layer, AI_STATIC, 
  .nl_params = &nl_5_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_4_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_4_layer, 4,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_4_chain,
  NULL, &nl_5_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_2_weights, &conv2d_2_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_2_scratch0, &conv2d_2_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_2_layer, 3,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_sssa8_ch_nl_pool,
  &conv2d_2_chain,
  NULL, &transpose_4_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_integer_INT8), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  transpose_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pad_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &transpose_1_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  transpose_1_layer, 1,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &transpose_1_chain,
  NULL, &conv2d_2_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)


AI_STATIC_CONST ai_i8 pad_0_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    pad_0_value, AI_ARRAY_FORMAT_S8,
    pad_0_value_data, pad_0_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  pad_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &in_pad_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pad_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pad_0_layer, 0,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &pad_0_chain,
  NULL, &transpose_1_layer, AI_STATIC, 
  .value = &pad_0_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  in_pad_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_input0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &in_pad_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  in_pad_0_layer, 0,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &in_pad_0_chain,
  NULL, &pad_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 22284, 1, 1),
    22284, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 8384, 1, 1),
    8384, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_PLANTS_VILLAGE_PT_IN_NUM, &serving_default_input0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_PLANTS_VILLAGE_PT_OUT_NUM, &gemm_14_output),
  &in_pad_0_layer, 0x1e1cfd47, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 22284, 1, 1),
      22284, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 8384, 1, 1),
      8384, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_PLANTS_VILLAGE_PT_IN_NUM, &serving_default_input0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_PLANTS_VILLAGE_PT_OUT_NUM, &gemm_14_output),
  &in_pad_0_layer, 0x1e1cfd47, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool plants_village_pt_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_plants_village_pt_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    serving_default_input0_output_array.data = AI_PTR(g_plants_village_pt_activations_map[0] + 1980);
    serving_default_input0_output_array.data_start = AI_PTR(g_plants_village_pt_activations_map[0] + 1980);
    in_pad_0_output_array.data = AI_PTR(g_plants_village_pt_activations_map[0] + 2748);
    in_pad_0_output_array.data_start = AI_PTR(g_plants_village_pt_activations_map[0] + 2748);
    pad_0_output_array.data = AI_PTR(g_plants_village_pt_activations_map[0] + 2544);
    pad_0_output_array.data_start = AI_PTR(g_plants_village_pt_activations_map[0] + 2544);
    transpose_1_output_array.data = AI_PTR(g_plants_village_pt_activations_map[0] + 1568);
    transpose_1_output_array.data_start = AI_PTR(g_plants_village_pt_activations_map[0] + 1568);
    conv2d_2_scratch0_array.data = AI_PTR(g_plants_village_pt_activations_map[0] + 2544);
    conv2d_2_scratch0_array.data_start = AI_PTR(g_plants_village_pt_activations_map[0] + 2544);
    conv2d_2_scratch1_array.data = AI_PTR(g_plants_village_pt_activations_map[0] + 3740);
    conv2d_2_scratch1_array.data_start = AI_PTR(g_plants_village_pt_activations_map[0] + 3740);
    conv2d_2_output_array.data = AI_PTR(g_plants_village_pt_activations_map[0] + 1280);
    conv2d_2_output_array.data_start = AI_PTR(g_plants_village_pt_activations_map[0] + 1280);
    transpose_4_output_array.data = AI_PTR(g_plants_village_pt_activations_map[0] + 2304);
    transpose_4_output_array.data_start = AI_PTR(g_plants_village_pt_activations_map[0] + 2304);
    nl_5_output_array.data = AI_PTR(g_plants_village_pt_activations_map[0] + 1280);
    nl_5_output_array.data_start = AI_PTR(g_plants_village_pt_activations_map[0] + 1280);
    in_pad_6_output_array.data = AI_PTR(g_plants_village_pt_activations_map[0] + 2304);
    in_pad_6_output_array.data_start = AI_PTR(g_plants_village_pt_activations_map[0] + 2304);
    pad_6_output_array.data = AI_PTR(g_plants_village_pt_activations_map[0] + 1728);
    pad_6_output_array.data_start = AI_PTR(g_plants_village_pt_activations_map[0] + 1728);
    transpose_7_output_array.data = AI_PTR(g_plants_village_pt_activations_map[0] + 128);
    transpose_7_output_array.data_start = AI_PTR(g_plants_village_pt_activations_map[0] + 128);
    conv2d_8_scratch0_array.data = AI_PTR(g_plants_village_pt_activations_map[0] + 1728);
    conv2d_8_scratch0_array.data_start = AI_PTR(g_plants_village_pt_activations_map[0] + 1728);
    conv2d_8_scratch1_array.data = AI_PTR(g_plants_village_pt_activations_map[0] + 7872);
    conv2d_8_scratch1_array.data_start = AI_PTR(g_plants_village_pt_activations_map[0] + 7872);
    conv2d_8_output_array.data = AI_PTR(g_plants_village_pt_activations_map[0] + 0);
    conv2d_8_output_array.data_start = AI_PTR(g_plants_village_pt_activations_map[0] + 0);
    transpose_10_output_array.data = AI_PTR(g_plants_village_pt_activations_map[0] + 512);
    transpose_10_output_array.data_start = AI_PTR(g_plants_village_pt_activations_map[0] + 512);
    nl_11_output_array.data = AI_PTR(g_plants_village_pt_activations_map[0] + 0);
    nl_11_output_array.data_start = AI_PTR(g_plants_village_pt_activations_map[0] + 0);
    gemm_13_scratch0_array.data = AI_PTR(g_plants_village_pt_activations_map[0] + 512);
    gemm_13_scratch0_array.data_start = AI_PTR(g_plants_village_pt_activations_map[0] + 512);
    gemm_13_output_array.data = AI_PTR(g_plants_village_pt_activations_map[0] + 1536);
    gemm_13_output_array.data_start = AI_PTR(g_plants_village_pt_activations_map[0] + 1536);
    gemm_14_scratch0_array.data = AI_PTR(g_plants_village_pt_activations_map[0] + 0);
    gemm_14_scratch0_array.data_start = AI_PTR(g_plants_village_pt_activations_map[0] + 0);
    gemm_14_output_array.data = AI_PTR(g_plants_village_pt_activations_map[0] + 64);
    gemm_14_output_array.data_start = AI_PTR(g_plants_village_pt_activations_map[0] + 64);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool plants_village_pt_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_plants_village_pt_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    conv2d_2_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_2_weights_array.data = AI_PTR(g_plants_village_pt_weights_map[0] + 0);
    conv2d_2_weights_array.data_start = AI_PTR(g_plants_village_pt_weights_map[0] + 0);
    conv2d_2_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_2_bias_array.data = AI_PTR(g_plants_village_pt_weights_map[0] + 432);
    conv2d_2_bias_array.data_start = AI_PTR(g_plants_village_pt_weights_map[0] + 432);
    conv2d_8_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_8_weights_array.data = AI_PTR(g_plants_village_pt_weights_map[0] + 496);
    conv2d_8_weights_array.data_start = AI_PTR(g_plants_village_pt_weights_map[0] + 496);
    conv2d_8_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_8_bias_array.data = AI_PTR(g_plants_village_pt_weights_map[0] + 5104);
    conv2d_8_bias_array.data_start = AI_PTR(g_plants_village_pt_weights_map[0] + 5104);
    gemm_13_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_13_weights_array.data = AI_PTR(g_plants_village_pt_weights_map[0] + 5232);
    gemm_13_weights_array.data_start = AI_PTR(g_plants_village_pt_weights_map[0] + 5232);
    gemm_13_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_13_bias_array.data = AI_PTR(g_plants_village_pt_weights_map[0] + 21616);
    gemm_13_bias_array.data_start = AI_PTR(g_plants_village_pt_weights_map[0] + 21616);
    gemm_14_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_14_weights_array.data = AI_PTR(g_plants_village_pt_weights_map[0] + 21744);
    gemm_14_weights_array.data_start = AI_PTR(g_plants_village_pt_weights_map[0] + 21744);
    gemm_14_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_14_bias_array.data = AI_PTR(g_plants_village_pt_weights_map[0] + 22224);
    gemm_14_bias_array.data_start = AI_PTR(g_plants_village_pt_weights_map[0] + 22224);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_plants_village_pt_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_PLANTS_VILLAGE_PT_MODEL_NAME,
      .model_signature   = AI_PLANTS_VILLAGE_PT_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 433093,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x1e1cfd47,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_plants_village_pt_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_PLANTS_VILLAGE_PT_MODEL_NAME,
      .model_signature   = AI_PLANTS_VILLAGE_PT_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 433093,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x1e1cfd47,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_plants_village_pt_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_plants_village_pt_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_plants_village_pt_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_plants_village_pt_create(network, AI_PLANTS_VILLAGE_PT_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_plants_village_pt_data_params_get(&params) != true) {
    err = ai_plants_village_pt_get_error(*network);
    return err;
  }
#if defined(AI_PLANTS_VILLAGE_PT_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_PLANTS_VILLAGE_PT_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_plants_village_pt_init(*network, &params) != true) {
    err = ai_plants_village_pt_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_plants_village_pt_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_plants_village_pt_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_plants_village_pt_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_plants_village_pt_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= plants_village_pt_configure_weights(net_ctx, params);
  ok &= plants_village_pt_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_plants_village_pt_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_plants_village_pt_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_PLANTS_VILLAGE_PT_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

