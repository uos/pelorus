CC ?= gcc
CFLAGS ?= -Wall -Wextra -I..
TARGET_DIR ?= ../../target/debug
LDFLAGS ?= -L$(TARGET_DIR) -lpelorus -lpthread -ldl -lm

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    LDFLAGS += -Wl,-rpath,$(TARGET_DIR)
endif
ifeq ($(UNAME_S),Darwin)
    LDFLAGS += -Wl,-rpath,$(TARGET_DIR)
endif

EXAMPLES = basic_usage

all: $(EXAMPLES)

basic_usage: basic_usage.c ../pelorus.h
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

clean:
	rm -f $(EXAMPLES)

run: basic_usage
	./basic_usage

.PHONY: all clean run
