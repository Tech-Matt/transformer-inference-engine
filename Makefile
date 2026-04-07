SRC_DIR=src/
INC_DIR=include/
EXECUTABLE_NAME=engine
FLAGS=-Wall -Wextra -Werror -g
LINKER_FLAGS=-lm

all:
	@echo "Compiling..."
	gcc $(FLAGS) $(SRC_DIR)*.c -I $(INC_DIR)  -o $(EXECUTABLE_NAME) $(LINKER_FLAGS)
	@echo "Compilation over."

clean:
	rm -f $(EXECUTABLE_NAME)
