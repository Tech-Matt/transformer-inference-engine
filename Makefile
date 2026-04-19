SRC_DIR=src/
INC_DIR=include/
BUILD_DIR=build/
EXECUTABLE_NAME=engine
FLAGS=-Wall -Wextra -Werror -g
LINKER_FLAGS=-lm

all:
	@echo "Compiling..."
	gcc $(FLAGS) $(SRC_DIR)*.c -I $(INC_DIR)  -o $(BUILD_DIR)$(EXECUTABLE_NAME) $(LINKER_FLAGS)
	@echo "Compilation over."

clean:
	rm -f $(BUILD_DIR)$(EXECUTABLE_NAME)


TEST_EXECUTABLE_NAME=test_engine
TEST_DIR=tests/
UNITY_DIR=vendor/unity/

test:
	@echo "Compiling and running ops tests..."
	gcc $(FLAGS) \
		$(TEST_DIR)test_ops.c \
		$(SRC_DIR)ops.c \
		$(SRC_DIR)tensor.c \
		$(UNITY_DIR)unity.c \
		-I $(INC_DIR) \
		-I $(UNITY_DIR) \
		-o $(BUILD_DIR)test_ops_bin \
		$(LINKER_FLAGS)
	./$(BUILD_DIR)test_ops_bin

	@echo "Compiling and running tensor tests..."
	gcc $(FLAGS) \
		$(TEST_DIR)test_tensor.c \
		$(SRC_DIR)tensor.c \
		$(UNITY_DIR)unity.c \
		-I $(INC_DIR) \
		-I $(UNITY_DIR) \
		-o $(BUILD_DIR)test_tensor_bin \
		$(LINKER_FLAGS)
	./$(BUILD_DIR)test_tensor_bin

clean-test:
	rm -f test_ops_bin test_tensor_bin