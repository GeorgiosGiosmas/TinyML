/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
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
#include "usb_host.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>

#include "ai_datatypes_defines.h"
#include "ai_platform.h"
#include "plantsvillage.h"
#include "plantsvillage_data.h"

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
I2C_HandleTypeDef hi2c1;

I2S_HandleTypeDef hi2s3;

SPI_HandleTypeDef hspi1;

TIM_HandleTypeDef htim10;

UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_I2C1_Init(void);
static void MX_I2S3_Init(void);
static void MX_SPI1_Init(void);
static void MX_TIM10_Init(void);
static void MX_USART2_UART_Init(void);
void MX_USB_HOST_Process(void);

/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */
  char buf[50];
  int buf_len = 0;
  ai_error ai_err;
  ai_i32 nbatch;
  uint32_t timestamp;
  float y_val;

  // Chunk of memory used to hold intermediate values for neural network
  AI_ALIGNED(4) ai_u8 activations[AI_PLANTSVILLAGE_DATA_ACTIVATIONS_SIZE];

  // Buffers used to store input and output tensors
  AI_ALIGNED(4) ai_i8 in_data[AI_PLANTSVILLAGE_IN_1_SIZE_BYTES];
  AI_ALIGNED(4) ai_i8 out_data[AI_PLANTSVILLAGE_OUT_1_SIZE_BYTES];

  // Pointer to our model
  ai_handle plantsvillage_model = AI_HANDLE_NULL;

  // Initialize wrapper structs that hold pointers to data and info about
  // data (tensor height, width, channels)
  ai_buffer ai_input[AI_PLANTSVILLAGE_IN_NUM] = { 0 };
  ai_buffer ai_output[AI_PLANTSVILLAGE_OUT_NUM] = { 0 };

  // Set working memory and get weights/biases from model
  ai_network_params ai_params = {
		  AI_PLANTSVILLAGE_DATA_WEIGHTS(ai_plantsvillage_data_weights_get()),
		  AI_PLANTSVILLAGE_DATA_ACTIVATIONS(activations)
  };

  // Set pointers wrapper structs to our data buffers
  ai_input[0].format = 1;
  ai_input[0].data = AI_HANDLE_PTR(in_data);
  ai_output[0].format = 1;
  ai_output[0].data = AI_HANDLE_PTR(out_data);

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
  MX_I2C1_Init();
  MX_I2S3_Init();
  MX_SPI1_Init();
  MX_USB_HOST_Init();
  MX_TIM10_Init();
  MX_USART2_UART_Init();
  /* USER CODE BEGIN 2 */

  // Start timer/counter
  HAL_TIM_Base_Start(&htim10);

  // Greetings
  buf_len = sprintf(buf, "\r\n\r\nSTM32 X-Cube-AI test\r\n");
  HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);

  // Create an instance of the neural network
  ai_err = ai_plantsvillage_create(&plantsvillage_model, AI_PLANTSVILLAGE_DATA_CONFIG);
  if(ai_err.type != AI_ERROR_NONE)
  {
	  buf_len = sprintf(buf, "Error: couldn't create NN instance\r\n");
	  HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
	  while(1);
  }

  // Initialize Neural Network
  if(!ai_plantsvillage_init(plantsvillage_model, &ai_params))
  {
	  buf_len = sprintf(buf, "Error: couldn't initialize NN\r\n");
	  HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
	  while(1);
  }

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
	// Fill input buffer
	/*(ai_float *)in_data = {
			0xc2b8b6, 0xc1b7b5, 0xc1b8b3, 0xbbb0ac, 0xb4a9a5, 0xafa4a0, 0xafa4a2, 0xb8aca9, 0xbaaead, 0xbaafad, 0xbaafad, 0xb6aba7, 0xb0a29f, 0xaba09c, 0xaca19d, 0xb3a8a6, 0xb8ada9, 0xb9aeaa, 0xb6aba7, 0xb1a6a2, 0xaca19d, 0xada29e, 0xbcadad, 0x827e6f, 0x494f36, 0xaea69f, 0xb8acaa, 0xaca29f, 0xaa9f9b, 0xa99b98, 0xada29e, 0xb4a9a5, 0xb5aaa6, 0xb3a8a4, 0xb1a6a2, 0xb1a6a2, 0xb4aaa5, 0xbdb1af, 0x9d978d, 0x3e4b31, 0x394b31, 0x66705b, 0xb2a6a3, 0xafa19e, 0xaa9c99, 0xab9d9a, 0xad9f9c, 0xac9f9c, 0xad9f9c, 0xafa19e, 0xaea09d, 0xafa4a0, 0xb7a9a7, 0xa19991, 0x44553d, 0x425640, 0x506050, 0x364932, 0x7b7f6d, 0xb4a5a3, 0xa89d9b, 0xa99b98, 0xa79998, 0xa79996, 0xad9f9c, 0xab9d9a, 0xab9d9a, 0xa79a96, 0xb0a09e, 0x606650, 0x314329, 0x364732, 0x4d5d4f, 0x435643, 0x3d4e38, 0x9b908b, 0xac9d9b, 0xa39694, 0x9f918e, 0xa29491, 0xa99b98, 0xa49693, 0xa59894, 0xab9b9a, 0x878073, 0x36462b, 0x3c4e35, 0x3e4d36, 0x516351, 0x3d503b, 0x364831, 0x656157, 0xa69795, 0x9d8f8b, 0xa0918e, 0xa29493, 0xa49591, 0xa29390, 0xa79895, 0xab9c99, 0x6c745e, 0x3a4f33, 0x3c4b35, 0x465541, 0x4f614b, 0x495c44, 0x293724, 0x393b2f, 0x9a8b89, 0x9e8f8c, 0xa09090, 0x9c8c8c, 0x9d8e88, 0xa39491, 0xa1928f, 0x9c8d87, 0x5e6245, 0x4a5c3b, 0x48583f, 0x5a6a54, 0x505f46, 0x2f3a2b, 0x2c3c2b, 0x3a3d30, 0x8c7b79, 0x9a8b88, 0x978787, 0x998a87, 0x9b8a83, 0x9c8b84, 0x998881, 0x9a8882, 0x4f4c34, 0x232b10, 0x37442d, 0x5d6b51, 0x49563f, 0x475844, 0x3f523c, 0x4b5444, 0x82716f, 0x978583, 0x948484, 0x938383, 0x8f7d7a, 0x96857e, 0x96857e, 0x958280, 0x83756c, 0x3d4129, 0x313c24, 0x55614a, 0x5b6d53, 0x536750, 0x485b47, 0x40523f, 0x5f6253, 0x92837e, 0x8f7f7d, 0x8c7a7a, 0x907e78, 0x94837c, 0x94837c, 0x91807a, 0x97857e, 0x675952, 0x252219, 0x4e573c, 0x56694b, 0x4a5e45, 0x52664f, 0x51664f, 0x36382b, 0x5d4d4a, 0x8e7c79, 0x867472, 0x8e7c76, 0x93817b, 0x93817b, 0x907f7b, 0x93827b, 0x8a7671, 0x615047, 0x424529, 0x3c4730, 0x343b29, 0x2f3325, 0x201e15, 0x1d1111, 0x5d4c49, 0x897770, 0x857170, 0x8f7c76, 0x8e7b75, 0x8e7b75, 0x8a7772, 0x8b7872, 0x93807b, 0x806e62, 0x312719, 0x22190e, 0x1b170a, 0x1c130d, 0x1b0f0f, 0x3c2c2b, 0x83726c, 0x8a7972, 0x826e6d, 0x8a7771, 0x8c7973, 0x8c7973, 0x897670, 0x8a7771, 0x8c7973, 0x8c7974, 0x786660, 0x63524c, 0x4d3f3a, 0x575040, 0x554641, 0x776462, 0x87766f, 0x84716b, 0x847270, 0x88736e, 0x8a7772, 0x88746f, 0x84716b, 0x86726d, 0x86736d, 0x897670, 0x8a7771, 0x8a7670, 0x846f6a, 0x756a5a, 0x73645e, 0x877573, 0x806c6b, 0x816f6d, 0x7f6e6c, 0x86716c, 0x816c67, 0x826d68, 0x806d67, 0x826d68, 0x83706a, 0x816f69, 0x7c6963, 0x7b6862, 0x7b6862, 0x76675c, 0x655450, 0x7a6867, 0x847270, 0x7e6c69, 0x7a6965
  	};*/
	for(uint32_t i = 0; i <AI_PLANTSVILLAGE_IN_1_SIZE; i++)
	{
		((ai_float *)in_data)[i] = (ai_float)2.0;
	}

	// Get timer timestamp
	timestamp = htim10.Instance->CNT;

	// Permorm Inference
	nbatch = ai_plantsvillage_run(plantsvillage_model, &ai_input, &ai_output);
	if(nbatch!=1)
	{
	  buf_len = sprintf(buf, "Error: couldn't run inference\r\n");
	  HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
	}

	// Read output (predicted y) of neural network
	y_val = ((float *)out_data)[0];

	// Print output of neural network along with inference time (microseconds)
	buf_len = sprintf(buf,
					  "Output: %f | Duration: %lu\r\n",
					  y_val,
					  htim10.Instance->CNT - timestamp);

	HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);

	// Wait before doing it again
	HAL_Delay(500);

    /* USER CODE END WHILE */
    MX_USB_HOST_Process();

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
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV2;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief I2C1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2C1_Init(void)
{

  /* USER CODE BEGIN I2C1_Init 0 */

  /* USER CODE END I2C1_Init 0 */

  /* USER CODE BEGIN I2C1_Init 1 */

  /* USER CODE END I2C1_Init 1 */
  hi2c1.Instance = I2C1;
  hi2c1.Init.ClockSpeed = 100000;
  hi2c1.Init.DutyCycle = I2C_DUTYCYCLE_2;
  hi2c1.Init.OwnAddress1 = 0;
  hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c1.Init.OwnAddress2 = 0;
  hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2C1_Init 2 */

  /* USER CODE END I2C1_Init 2 */

}

/**
  * @brief I2S3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2S3_Init(void)
{

  /* USER CODE BEGIN I2S3_Init 0 */

  /* USER CODE END I2S3_Init 0 */

  /* USER CODE BEGIN I2S3_Init 1 */

  /* USER CODE END I2S3_Init 1 */
  hi2s3.Instance = SPI3;
  hi2s3.Init.Mode = I2S_MODE_MASTER_TX;
  hi2s3.Init.Standard = I2S_STANDARD_PHILIPS;
  hi2s3.Init.DataFormat = I2S_DATAFORMAT_16B;
  hi2s3.Init.MCLKOutput = I2S_MCLKOUTPUT_ENABLE;
  hi2s3.Init.AudioFreq = I2S_AUDIOFREQ_96K;
  hi2s3.Init.CPOL = I2S_CPOL_LOW;
  hi2s3.Init.ClockSource = I2S_CLOCK_PLL;
  hi2s3.Init.FullDuplexMode = I2S_FULLDUPLEXMODE_DISABLE;
  if (HAL_I2S_Init(&hi2s3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2S3_Init 2 */

  /* USER CODE END I2S3_Init 2 */

}

/**
  * @brief SPI1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_SPI1_Init(void)
{

  /* USER CODE BEGIN SPI1_Init 0 */

  /* USER CODE END SPI1_Init 0 */

  /* USER CODE BEGIN SPI1_Init 1 */

  /* USER CODE END SPI1_Init 1 */
  /* SPI1 parameter configuration*/
  hspi1.Instance = SPI1;
  hspi1.Init.Mode = SPI_MODE_MASTER;
  hspi1.Init.Direction = SPI_DIRECTION_2LINES;
  hspi1.Init.DataSize = SPI_DATASIZE_8BIT;
  hspi1.Init.CLKPolarity = SPI_POLARITY_LOW;
  hspi1.Init.CLKPhase = SPI_PHASE_1EDGE;
  hspi1.Init.NSS = SPI_NSS_SOFT;
  hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_2;
  hspi1.Init.FirstBit = SPI_FIRSTBIT_MSB;
  hspi1.Init.TIMode = SPI_TIMODE_DISABLE;
  hspi1.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
  hspi1.Init.CRCPolynomial = 10;
  if (HAL_SPI_Init(&hspi1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN SPI1_Init 2 */

  /* USER CODE END SPI1_Init 2 */

}

/**
  * @brief TIM10 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM10_Init(void)
{

  /* USER CODE BEGIN TIM10_Init 0 */

  /* USER CODE END TIM10_Init 0 */

  /* USER CODE BEGIN TIM10_Init 1 */

  /* USER CODE END TIM10_Init 1 */
  htim10.Instance = TIM10;
  htim10.Init.Prescaler = 80-1;
  htim10.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim10.Init.Period = 65535 - 1;
  htim10.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim10.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim10) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM10_Init 2 */

  /* USER CODE END TIM10_Init 2 */

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

  /*Configure GPIO pin : OTG_FS_OverCurrent_Pin */
  GPIO_InitStruct.Pin = OTG_FS_OverCurrent_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(OTG_FS_OverCurrent_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : MEMS_INT2_Pin */
  GPIO_InitStruct.Pin = MEMS_INT2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_EVT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(MEMS_INT2_GPIO_Port, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

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

#ifdef  USE_FULL_ASSERT
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
