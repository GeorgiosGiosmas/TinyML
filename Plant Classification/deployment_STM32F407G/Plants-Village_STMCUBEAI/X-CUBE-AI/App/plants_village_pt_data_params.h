/**
  ******************************************************************************
  * @file    plants_village_pt_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-03-20T15:16:33+0200
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#ifndef PLANTS_VILLAGE_PT_DATA_PARAMS_H
#define PLANTS_VILLAGE_PT_DATA_PARAMS_H

#include "ai_platform.h"

/*
#define AI_PLANTS_VILLAGE_PT_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_plants_village_pt_data_weights_params[1]))
*/

#define AI_PLANTS_VILLAGE_PT_DATA_CONFIG               (NULL)


#define AI_PLANTS_VILLAGE_PT_DATA_ACTIVATIONS_SIZES \
  { 8384, }
#define AI_PLANTS_VILLAGE_PT_DATA_ACTIVATIONS_SIZE     (8384)
#define AI_PLANTS_VILLAGE_PT_DATA_ACTIVATIONS_COUNT    (1)
#define AI_PLANTS_VILLAGE_PT_DATA_ACTIVATION_1_SIZE    (8384)



#define AI_PLANTS_VILLAGE_PT_DATA_WEIGHTS_SIZES \
  { 22284, }
#define AI_PLANTS_VILLAGE_PT_DATA_WEIGHTS_SIZE         (22284)
#define AI_PLANTS_VILLAGE_PT_DATA_WEIGHTS_COUNT        (1)
#define AI_PLANTS_VILLAGE_PT_DATA_WEIGHT_1_SIZE        (22284)



#define AI_PLANTS_VILLAGE_PT_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_plants_village_pt_activations_table[1])

extern ai_handle g_plants_village_pt_activations_table[1 + 2];



#define AI_PLANTS_VILLAGE_PT_DATA_WEIGHTS_TABLE_GET() \
  (&g_plants_village_pt_weights_table[1])

extern ai_handle g_plants_village_pt_weights_table[1 + 2];


#endif    /* PLANTS_VILLAGE_PT_DATA_PARAMS_H */
