################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/edge-impulse-sdk/porting/espressif/ESP-NN/src/basic_math/esp_nn_add_ansi.c \
../Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/edge-impulse-sdk/porting/espressif/ESP-NN/src/basic_math/esp_nn_mul_ansi.c 

C_DEPS += \
./Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/edge-impulse-sdk/porting/espressif/ESP-NN/src/basic_math/esp_nn_add_ansi.d \
./Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/edge-impulse-sdk/porting/espressif/ESP-NN/src/basic_math/esp_nn_mul_ansi.d 

OBJS += \
./Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/edge-impulse-sdk/porting/espressif/ESP-NN/src/basic_math/esp_nn_add_ansi.o \
./Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/edge-impulse-sdk/porting/espressif/ESP-NN/src/basic_math/esp_nn_mul_ansi.o 


# Each subdirectory must supply rules for building sources it contributes
Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/edge-impulse-sdk/porting/espressif/ESP-NN/src/basic_math/%.o Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/edge-impulse-sdk/porting/espressif/ESP-NN/src/basic_math/%.su Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/edge-impulse-sdk/porting/espressif/ESP-NN/src/basic_math/%.cyclo: ../Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/edge-impulse-sdk/porting/espressif/ESP-NN/src/basic_math/%.c Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/edge-impulse-sdk/porting/espressif/ESP-NN/src/basic_math/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -DUSE_HAL_DRIVER -DSTM32F407xx -c -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -I../Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/ -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Middlewares-2f-Third_Party-2f-EdgeImpulse_plant_village_classification_MachineLearning-2f-edgeimpulse-2f-edge-2d-impulse-2d-sdk-2f-porting-2f-espressif-2f-ESP-2d-NN-2f-src-2f-basic_math

clean-Middlewares-2f-Third_Party-2f-EdgeImpulse_plant_village_classification_MachineLearning-2f-edgeimpulse-2f-edge-2d-impulse-2d-sdk-2f-porting-2f-espressif-2f-ESP-2d-NN-2f-src-2f-basic_math:
	-$(RM) ./Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/edge-impulse-sdk/porting/espressif/ESP-NN/src/basic_math/esp_nn_add_ansi.cyclo ./Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/edge-impulse-sdk/porting/espressif/ESP-NN/src/basic_math/esp_nn_add_ansi.d ./Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/edge-impulse-sdk/porting/espressif/ESP-NN/src/basic_math/esp_nn_add_ansi.o ./Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/edge-impulse-sdk/porting/espressif/ESP-NN/src/basic_math/esp_nn_add_ansi.su ./Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/edge-impulse-sdk/porting/espressif/ESP-NN/src/basic_math/esp_nn_mul_ansi.cyclo ./Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/edge-impulse-sdk/porting/espressif/ESP-NN/src/basic_math/esp_nn_mul_ansi.d ./Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/edge-impulse-sdk/porting/espressif/ESP-NN/src/basic_math/esp_nn_mul_ansi.o ./Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/edge-impulse-sdk/porting/espressif/ESP-NN/src/basic_math/esp_nn_mul_ansi.su

.PHONY: clean-Middlewares-2f-Third_Party-2f-EdgeImpulse_plant_village_classification_MachineLearning-2f-edgeimpulse-2f-edge-2d-impulse-2d-sdk-2f-porting-2f-espressif-2f-ESP-2d-NN-2f-src-2f-basic_math

