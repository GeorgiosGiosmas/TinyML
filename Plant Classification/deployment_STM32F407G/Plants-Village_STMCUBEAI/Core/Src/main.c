/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
CRC_HandleTypeDef hcrc;

TIM_HandleTypeDef htim6;

UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_CRC_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_TIM6_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/**
* @brief Statically allocated buffers.
* Buffers can be dynamically allocated using malloc and size information
* given by the report in ai_network_get_report().
*/
ai_i8 activations[AI_PLANTS_VILLAGE_PT_DATA_ACTIVATIONS_SIZE];
ai_i8 in_data[AI_PLANTS_VILLAGE_PT_IN_1_SIZE_BYTES];
ai_i8 out_data[AI_PLANTS_VILLAGE_PT_OUT_1_SIZE_BYTES];

/* AI buffer IO handlers */
ai_buffer *ai_input;
ai_buffer *ai_output;

/* Input Data */
const float input_data[768] = {
		0.5373, 0.5020, 0.5843, 0.5412, 0.5059, 0.5882, 0.5451, 0.5098, 0.5922, 0.5490, 0.5137, 0.5961, 0.5490, 0.5137, 0.5961, 0.5569, 0.5216, 0.6039, 0.5529, 0.5176, 0.6000, 0.5451, 0.5098, 0.5882, 0.5490, 0.5137, 0.6039, 0.5490, 0.5059, 0.6157, 0.5373, 0.4941, 0.6000, 0.5451, 0.5059, 0.5882, 0.5412, 0.5059, 0.5882, 0.5373, 0.5020, 0.5843, 0.5333, 0.4941, 0.5804, 0.5294, 0.4902, 0.5843, 0.5294, 0.4941, 0.5765, 0.5333, 0.4980, 0.5804, 0.5373, 0.5020, 0.5843, 0.5451, 0.5098, 0.5922, 0.5451, 0.5098, 0.5922, 0.5451, 0.5098, 0.5922, 0.5490, 0.5137, 0.5843, 0.5529, 0.5137, 0.6078, 0.5490, 0.5137, 0.5765, 0.5412, 0.5412, 0.4392, 0.4902, 0.5059, 0.3255, 0.5255, 0.4941, 0.5451, 0.5451, 0.5098, 0.5961, 0.5412, 0.5059, 0.5882, 0.5412, 0.5059, 0.5882, 0.5333, 0.4980, 0.5804, 0.5333, 0.4980, 0.5804, 0.5333, 0.4980, 0.5804, 0.5373, 0.5020, 0.5843, 0.5373, 0.5059, 0.5843, 0.5412, 0.5059, 0.5804, 0.5451, 0.5059, 0.5922, 0.5569, 0.5098, 0.6157, 0.5216, 0.5216, 0.4941, 0.4588, 0.5490, 0.2549, 0.4549, 0.5686, 0.1922, 0.4471, 0.5333, 0.1961, 0.4980, 0.4706, 0.5137, 0.5333, 0.4941, 0.5961, 0.5333, 0.4980, 0.5765, 0.5373, 0.5020, 0.5843, 0.5373, 0.5020, 0.5843, 0.5294, 0.4941, 0.5765, 0.5294, 0.4941, 0.5765, 0.5294, 0.4941, 0.5765, 0.5451, 0.5020, 0.6000, 0.5529, 0.5020, 0.6157, 0.5216, 0.5137, 0.5373, 0.4706, 0.5490, 0.4039, 0.4549, 0.5765, 0.2784, 0.4314, 0.5569, 0.2157, 0.4863, 0.5647, 0.2039, 0.4431, 0.4549, 0.1843, 0.4510, 0.4392, 0.3765, 0.5412, 0.5020, 0.5020, 0.5333, 0.4941, 0.5804, 0.5333, 0.4980, 0.5686, 0.5294, 0.4941, 0.5725, 0.5294, 0.4941, 0.5765, 0.5294, 0.4941, 0.5765, 0.5255, 0.4902, 0.5725, 0.4549, 0.5020, 0.4353, 0.4627, 0.4784, 0.4471, 0.4588, 0.5137, 0.4000, 0.3961, 0.5765, 0.2431, 0.4431, 0.5765, 0.2706, 0.4549, 0.5608, 0.2471, 0.4824, 0.5569, 0.1882, 0.4784, 0.5608, 0.1922, 0.4941, 0.5294, 0.1804, 0.5333, 0.5137, 0.4000, 0.5176, 0.4784, 0.5765, 0.5373, 0.5020, 0.5686, 0.5294, 0.4941, 0.5765, 0.5294, 0.4941, 0.5765, 0.5255, 0.4863, 0.5686, 0.5098, 0.4745, 0.5529, 0.3882, 0.5098, 0.2980, 0.3373, 0.4980, 0.2000, 0.3529, 0.4863, 0.2392, 0.4431, 0.5569, 0.2549, 0.4235, 0.5569, 0.2549, 0.4353, 0.5529, 0.2275, 0.4706, 0.5529, 0.2078, 0.4706, 0.5843, 0.2196, 0.4157, 0.5020, 0.1647, 0.4588, 0.4471, 0.3255, 0.5059, 0.4667, 0.5333, 0.5137, 0.4784, 0.5529, 0.5137, 0.4784, 0.5608, 0.5216, 0.4863, 0.5647, 0.5098, 0.4745, 0.5490, 0.4863, 0.4627, 0.5255, 0.4000, 0.5137, 0.3137, 0.3569, 0.5137, 0.2314, 0.3333, 0.4863, 0.2157, 0.4078, 0.5216, 0.2392, 0.4314, 0.5490, 0.2471, 0.4235, 0.5451, 0.2118, 0.4667, 0.5412, 0.1725, 0.5216, 0.5882, 0.1882, 0.5490, 0.5529, 0.1490, 0.5451, 0.5176, 0.2000, 0.4902, 0.4588, 0.5176, 0.5059, 0.4706, 0.5490, 0.5176, 0.4824, 0.5529, 0.5098, 0.4745, 0.5490, 0.5137, 0.4706, 0.5569, 0.4667, 0.4314, 0.5059, 0.3608, 0.4392, 0.3098, 0.3804, 0.5333, 0.2392, 0.3490, 0.5137, 0.2275, 0.3216, 0.4902, 0.2078, 0.3490, 0.4980, 0.2157, 0.4510, 0.5373, 0.2039, 0.5686, 0.5294, 0.2353, 0.4980, 0.4353, 0.1608, 0.5059, 0.4588, 0.2157, 0.4353, 0.3922, 0.3882, 0.4745, 0.4431, 0.5176, 0.4902, 0.4549, 0.5294, 0.4980, 0.4627, 0.5373, 0.4980, 0.4588, 0.5412, 0.4118, 0.4471, 0.3961, 0.2902, 0.4118, 0.2353, 0.2235, 0.3608, 0.1647, 0.3373, 0.4314, 0.1804, 0.3765, 0.4941, 0.1922, 0.3373, 0.4824, 0.2000, 0.3490, 0.4863, 0.1922, 0.4824, 0.5176, 0.2588, 0.6235, 0.4745, 0.3961, 0.4275, 0.3490, 0.2000, 0.3255, 0.2863, 0.2980, 0.3882, 0.3373, 0.4275, 0.4706, 0.4353, 0.4980, 0.4863, 0.4510, 0.5255, 0.4863, 0.4510, 0.5255, 0.4863, 0.4392, 0.5373, 0.3804, 0.4667, 0.3373, 0.3137, 0.5020, 0.2078, 0.3098, 0.4667, 0.2078, 0.2627, 0.4196, 0.1647, 0.2667, 0.3922, 0.1373, 0.3020, 0.4392, 0.1882, 0.3451, 0.4941, 0.1961, 0.4510, 0.4941, 0.2157, 0.5882, 0.4980, 0.2235, 0.4588, 0.4314, 0.2667, 0.3412, 0.3137, 0.3804, 0.4157, 0.3804, 0.4392, 0.4667, 0.4314, 0.4980, 0.4824, 0.4471, 0.5216, 0.4784, 0.4431, 0.5176, 0.4627, 0.4196, 0.5137, 0.3804, 0.3843, 0.3961, 0.2980, 0.4078, 0.2196, 0.3255, 0.4745, 0.2196, 0.3451, 0.5098, 0.2353, 0.3098, 0.4745, 0.2078, 0.3176, 0.4667, 0.2039, 0.3294, 0.4706, 0.1961, 0.3961, 0.4784, 0.1882, 0.3569, 0.3608, 0.2157, 0.3412, 0.3020, 0.3529, 0.3961, 0.3608, 0.4275, 0.4549, 0.4235, 0.4902, 0.4627, 0.4275, 0.5020, 0.4706, 0.4353, 0.5098, 0.4784, 0.4431, 0.5176, 0.4510, 0.4157, 0.4941, 0.4157, 0.3765, 0.4667, 0.3176, 0.3020, 0.3451, 0.1882, 0.2627, 0.1294, 0.1961, 0.3020, 0.0980, 0.2039, 0.2824, 0.0902, 0.2863, 0.4039, 0.1686, 0.3216, 0.4667, 0.1843, 0.3529, 0.4549, 0.1765, 0.3451, 0.3255, 0.2941, 0.3216, 0.2824, 0.3843, 0.4078, 0.3804, 0.4431, 0.4431, 0.4078, 0.4824, 0.4471, 0.4118, 0.4863, 0.4510, 0.4157, 0.4902, 0.4549, 0.4196, 0.4941, 0.4314, 0.3961, 0.4745, 0.4275, 0.4000, 0.4706, 0.4196, 0.3765, 0.4667, 0.3020, 0.3333, 0.2863, 0.2471, 0.3725, 0.1098, 0.2980, 0.4000, 0.1412, 0.3451, 0.4784, 0.2039, 0.3686, 0.5373, 0.2275, 0.2902, 0.3765, 0.1843, 0.3569, 0.3608, 0.2235, 0.3725, 0.3451, 0.3725, 0.4039, 0.3686, 0.4588, 0.4392, 0.4039, 0.4784, 0.4392, 0.4039, 0.4784, 0.4431, 0.4078, 0.4824, 0.4471, 0.4118, 0.4863, 0.4196, 0.3843, 0.4627, 0.4118, 0.3804, 0.4510, 0.4235, 0.3843, 0.4667, 0.3765, 0.3569, 0.3961, 0.2902, 0.3647, 0.2353, 0.3137, 0.4431, 0.2157, 0.3294, 0.4667, 0.2039, 0.2863, 0.3882, 0.1922, 0.2941, 0.2745, 0.3137, 0.3569, 0.3255, 0.3451, 0.4078, 0.3725, 0.3765, 0.4196, 0.3843, 0.4745, 0.4392, 0.4039, 0.4824, 0.4353, 0.4000, 0.4824, 0.4431, 0.4078, 0.4902, 0.4471, 0.4118, 0.4941, 0.4118, 0.3765, 0.4510, 0.4157, 0.3804, 0.4549, 0.4157, 0.3804, 0.4549, 0.4078, 0.3725, 0.4549, 0.3529, 0.3098, 0.4000, 0.2863, 0.2588, 0.3176, 0.2667, 0.2431, 0.2902, 0.2863, 0.2510, 0.3294, 0.3882, 0.3529, 0.4314, 0.4275, 0.3882, 0.4706, 0.4118, 0.3765, 0.4627, 0.4196, 0.3843, 0.4667, 0.4196, 0.3843, 0.4667, 0.4314, 0.3961, 0.4784, 0.4275, 0.3922, 0.4745, 0.4353, 0.4000, 0.4824, 0.4000, 0.3647, 0.4392, 0.4118, 0.3765, 0.4510, 0.4157, 0.3804, 0.4549, 0.4157, 0.3804, 0.4549, 0.4157, 0.3843, 0.4549, 0.4078, 0.3686, 0.4471, 0.4039, 0.3686, 0.4471, 0.4157, 0.3843, 0.4549, 0.4157, 0.3843, 0.4588, 0.4196, 0.3843, 0.4627, 0.4157, 0.3765, 0.4510, 0.4118, 0.3765, 0.4588, 0.4157, 0.3843, 0.4667, 0.4235, 0.3882, 0.4706, 0.4196, 0.3843, 0.4667, 0.4157, 0.3804, 0.4627
};

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

	/* USER CODE BEGIN 1 */

	char buffer[100] = { 0 };
    ai_handle network = AI_HANDLE_NULL;
    ai_error err;
    ai_network_report report;
    const int in_zp = -128;
    const int out_zp = 57;
    const double in_scale = 0.003613994689658284;
    const double out_scale = 0.5393611788749695;

	/* USER CODE END 1 */

	/* MCU Configuration--------------------------------------------------------*/

	/* Reset of all peripherals, Initializes the Flash interface and the Systick. */
	HAL_Init();

	/* USER CODE BEGIN Init */

	/* USER CODE END Init */

	/* Configure the system clock */
	SystemClock_Config();

	/* USER CODE BEGIN SysInit */

	/* USER CODE END SysInit */

	/* Initialize all configured peripherals */
	MX_GPIO_Init();
	MX_CRC_Init();
	MX_USART2_UART_Init();
	MX_TIM6_Init();
	/* USER CODE BEGIN 2 */

	/** @brief Initialize network */
	const ai_handle acts[] = { activations };
	err = ai_plants_village_pt_create_and_init(&network, acts, NULL);
	if (err.type != AI_ERROR_NONE) {
		sprintf(buffer, "ai init_and_create error\n\r");
		ai_print(&huart2, buffer, sizeof(buffer));

		return -1;
	}

	/** @brief {optional} for debug/log purpose */
	if (ai_plants_village_pt_get_report(network, &report) != true) {
		sprintf(buffer, "ai get report error\n\r");
		ai_print(&huart2, buffer, sizeof(buffer));

		return -1;
	}

	sprintf(buffer, "Model name      : %s\n\r", report.model_name);
	ai_print(&huart2, buffer, sizeof(buffer));

	sprintf(buffer, "Model signature : %s\n\r", report.model_signature);
	ai_print(&huart2, buffer, sizeof(buffer));

	ai_input = &report.inputs[0];
	ai_output = &report.outputs[0];

	sprintf(buffer, "input[0] : (%d, %d, %d)\n\r", AI_BUFFER_SHAPE_ELEM(ai_input, AI_SHAPE_HEIGHT),
										AI_BUFFER_SHAPE_ELEM(ai_input, AI_SHAPE_WIDTH),
										AI_BUFFER_SHAPE_ELEM(ai_input, AI_SHAPE_CHANNEL));
	ai_print(&huart2, buffer, sizeof(buffer));

	sprintf(buffer, "output[0] : (%d, %d, %d)\n\r", AI_BUFFER_SHAPE_ELEM(ai_output, AI_SHAPE_HEIGHT),
										 AI_BUFFER_SHAPE_ELEM(ai_output, AI_SHAPE_WIDTH),
										 AI_BUFFER_SHAPE_ELEM(ai_output, AI_SHAPE_CHANNEL));
	ai_print(&huart2, buffer, sizeof(buffer));

	/** @brief Fill input buffer with the input image */
	srand(1);
	for (int i = 0; i < AI_PLANTS_VILLAGE_PT_IN_1_SIZE; i++) {
		in_data[i] = (ai_i8)(round(input_data[i] / in_scale) + in_zp);
	}

	/** @brief Normalize, convert and/or quantize inputs if necessary... */

	/** @brief Perform inference */
	ai_i32 n_batch;

	/** @brief Create the AI buffer IO handlers
	 *  @note  ai_inuput/ai_output are already initilaized after the
	 *         ai_network_get_report() call. This is just here to illustrate
	 *         the case where get_report() is not called.
	 */
	ai_input = ai_plants_village_pt_inputs_get(network, NULL);
	ai_output = ai_plants_village_pt_outputs_get(network, NULL);

	/** @brief Set input/output buffer addresses */
	ai_input[0].data = AI_HANDLE_PTR(in_data);
	ai_output[0].data = AI_HANDLE_PTR(out_data);

	HAL_TIM_Base_Start(&htim6);

	int start = HAL_GetTick();

	/** @brief Perform the inference */
	n_batch = ai_plants_village_pt_run(network, &ai_input[0], &ai_output[0]);
	if (n_batch != 1) {
		err = ai_plants_village_pt_get_error(network);
		sprintf(buffer, "ai run error %d, %d\n\r", err.type, err.code);
		ai_print(&huart2, buffer, sizeof(buffer));

	  return -1;
	}

	int duration = HAL_GetTick() - start;

	sprintf(buffer, "Inference Time: %d \n\r", duration);
	ai_print(&huart2, buffer, sizeof(buffer));

	/** @brief Post-process the output results/predictions and normalize, convert and/or de-quantize outputs if necessary...*/
	sprintf(buffer, "Inference output: [ ");
	ai_print(&huart2, buffer, sizeof(buffer));

	for(int i = 0; i < AI_PLANTS_VILLAGE_PT_OUT_1_SIZE; i++) {
		if(i == AI_PLANTS_VILLAGE_PT_OUT_1_SIZE - 1)
		{
			sprintf(buffer, "%2.2f", (out_data[i] - out_zp) * out_scale);
			ai_print(&huart2, buffer, sizeof(buffer));
		}
		else
		{
			sprintf(buffer, "%2.2f, ", (out_data[i] - out_zp) * out_scale);
			ai_print(&huart2, buffer, sizeof(buffer));
		}
	}

	sprintf(buffer, " ]\n\n\r");
	ai_print(&huart2, buffer, sizeof(buffer));

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
	while (1)
	{
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
	}
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 7;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief CRC Initialization Function
  * @param None
  * @retval None
  */
static void MX_CRC_Init(void)
{

  /* USER CODE BEGIN CRC_Init 0 */

  /* USER CODE END CRC_Init 0 */

  /* USER CODE BEGIN CRC_Init 1 */

  /* USER CODE END CRC_Init 1 */
  hcrc.Instance = CRC;
  if (HAL_CRC_Init(&hcrc) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN CRC_Init 2 */

  /* USER CODE END CRC_Init 2 */

}

/**
  * @brief TIM6 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM6_Init(void)
{

  /* USER CODE BEGIN TIM6_Init 0 */

  /* USER CODE END TIM6_Init 0 */

  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM6_Init 1 */

  /* USER CODE END TIM6_Init 1 */
  htim6.Instance = TIM6;
  htim6.Init.Prescaler = 0;
  htim6.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim6.Init.Period = 65535;
  htim6.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim6) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim6, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM6_Init 2 */

  /* USER CODE END TIM6_Init 2 */

}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(CS_I2C_SPI_GPIO_Port, CS_I2C_SPI_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(OTG_FS_PowerSwitchOn_GPIO_Port, OTG_FS_PowerSwitchOn_Pin, GPIO_PIN_SET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOD, LD4_Pin|LD3_Pin|LD5_Pin|LD6_Pin
                          |Audio_RST_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : CS_I2C_SPI_Pin */
  GPIO_InitStruct.Pin = CS_I2C_SPI_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(CS_I2C_SPI_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : OTG_FS_PowerSwitchOn_Pin */
  GPIO_InitStruct.Pin = OTG_FS_PowerSwitchOn_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(OTG_FS_PowerSwitchOn_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : PDM_OUT_Pin */
  GPIO_InitStruct.Pin = PDM_OUT_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF5_SPI2;
  HAL_GPIO_Init(PDM_OUT_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : B1_Pin */
  GPIO_InitStruct.Pin = B1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_EVT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(B1_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : I2S3_WS_Pin */
  GPIO_InitStruct.Pin = I2S3_WS_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF6_SPI3;
  HAL_GPIO_Init(I2S3_WS_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : SPI1_SCK_Pin SPI1_MISO_Pin SPI1_MOSI_Pin */
  GPIO_InitStruct.Pin = SPI1_SCK_Pin|SPI1_MISO_Pin|SPI1_MOSI_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF5_SPI1;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pin : BOOT1_Pin */
  GPIO_InitStruct.Pin = BOOT1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(BOOT1_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : CLK_IN_Pin */
  GPIO_InitStruct.Pin = CLK_IN_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF5_SPI2;
  HAL_GPIO_Init(CLK_IN_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : LD4_Pin LD3_Pin LD5_Pin LD6_Pin
                           Audio_RST_Pin */
  GPIO_InitStruct.Pin = LD4_Pin|LD3_Pin|LD5_Pin|LD6_Pin
                          |Audio_RST_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

  /*Configure GPIO pins : I2S3_MCK_Pin I2S3_SCK_Pin I2S3_SD_Pin */
  GPIO_InitStruct.Pin = I2S3_MCK_Pin|I2S3_SCK_Pin|I2S3_SD_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF6_SPI3;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  /*Configure GPIO pin : VBUS_FS_Pin */
  GPIO_InitStruct.Pin = VBUS_FS_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(VBUS_FS_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : OTG_FS_ID_Pin OTG_FS_DM_Pin OTG_FS_DP_Pin */
  GPIO_InitStruct.Pin = OTG_FS_ID_Pin|OTG_FS_DM_Pin|OTG_FS_DP_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF10_OTG_FS;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pin : OTG_FS_OverCurrent_Pin */
  GPIO_InitStruct.Pin = OTG_FS_OverCurrent_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(OTG_FS_OverCurrent_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : Audio_SCL_Pin Audio_SDA_Pin */
  GPIO_InitStruct.Pin = Audio_SCL_Pin|Audio_SDA_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_OD;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF4_I2C1;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pin : MEMS_INT2_Pin */
  GPIO_InitStruct.Pin = MEMS_INT2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_EVT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(MEMS_INT2_GPIO_Port, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */
void ai_print(UART_HandleTypeDef *uart, const char *buffer, uint32_t bufferSize)
{
	if(HAL_UART_Transmit(uart, (uint8_t *)buffer, bufferSize, HAL_MAX_DELAY) != HAL_OK)
		for(;;);
	memset(buffer, 0, bufferSize);
}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
