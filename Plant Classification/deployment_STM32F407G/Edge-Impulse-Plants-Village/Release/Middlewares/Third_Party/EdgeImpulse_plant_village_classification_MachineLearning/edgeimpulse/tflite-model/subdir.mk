################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/tflite-model/tflite_learn_821727_14_compiled.cpp 

OBJS += \
./Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/tflite-model/tflite_learn_821727_14_compiled.o 

CPP_DEPS += \
./Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/tflite-model/tflite_learn_821727_14_compiled.d 


# Each subdirectory must supply rules for building sources it contributes
Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/tflite-model/%.o Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/tflite-model/%.su Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/tflite-model/%.cyclo: ../Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/tflite-model/%.cpp Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/tflite-model/subdir.mk
	arm-none-eabi-g++ "$<" -mcpu=cortex-m4 -std=gnu++14 -DUSE_HAL_DRIVER -DSTM32F407xx -c -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -I../Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/ -Os -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -fno-use-cxa-atexit -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Middlewares-2f-Third_Party-2f-EdgeImpulse_plant_village_classification_MachineLearning-2f-edgeimpulse-2f-tflite-2d-model

clean-Middlewares-2f-Third_Party-2f-EdgeImpulse_plant_village_classification_MachineLearning-2f-edgeimpulse-2f-tflite-2d-model:
	-$(RM) ./Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/tflite-model/tflite_learn_821727_14_compiled.cyclo ./Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/tflite-model/tflite_learn_821727_14_compiled.d ./Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/tflite-model/tflite_learn_821727_14_compiled.o ./Middlewares/Third_Party/EdgeImpulse_plant_village_classification_MachineLearning/edgeimpulse/tflite-model/tflite_learn_821727_14_compiled.su

.PHONY: clean-Middlewares-2f-Third_Party-2f-EdgeImpulse_plant_village_classification_MachineLearning-2f-edgeimpulse-2f-tflite-2d-model

