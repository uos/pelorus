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

EXAMPLES = async_callback sync_blocking

all: $(EXAMPLES)

async_callback: async_callback.c ../pelorus.h
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

sync_blocking: sync_blocking.c ../pelorus.h
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

clean:
	rm -f $(EXAMPLES)

run: sync_blocking async_callback
	./sync_blocking
	./async_callback

.PHONY: all clean run
