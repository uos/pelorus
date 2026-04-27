CC ?= gcc
CFLAGS ?= -Wall -Wextra -I..
BUILD_TYPE ?= release
TARGET_DIR ?= ../../target/$(BUILD_TYPE)
LDFLAGS ?= -L$(TARGET_DIR) -lpelorus -lpthread -ldl -lm
PICO_STATICLIB ?= $(TARGET_DIR)/libpelorus_pico.a

ifeq ($(BUILD_TYPE),release)
    CARGO_PROFILE_ARG := --release
else
    CARGO_PROFILE_ARG :=
endif

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    LDFLAGS += -Wl,-rpath,$(TARGET_DIR)
endif
ifeq ($(UNAME_S),Darwin)
    LDFLAGS += -Wl,-rpath,$(TARGET_DIR)
endif

PICO_FLAGS = -I../../pelorus_pico/include

EXAMPLES = async_callback sync_blocking pico_equality cpp_wrapper
CPP_BUILD_DIR ?= build

all: $(EXAMPLES)

async_callback: async_callback.c ../pelorus.h
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

sync_blocking: sync_blocking.c ../pelorus.h
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

pico_equality: pico_equality.c ../pelorus.h ../../pelorus_pico/include/pelorus_pico.h FORCE
	cargo build -p pelorus_c $(CARGO_PROFILE_ARG)
	RUSTFLAGS="-C panic=abort" cargo build -p pelorus_pico $(CARGO_PROFILE_ARG)
	$(CC) $(CFLAGS) $(PICO_FLAGS) $< $(LDFLAGS) $(PICO_STATICLIB) -o $@

# Build C++ example using CMake so it can find PCL/Eigen and use proper flags
cpp_wrapper:
	@echo "Configuring and building cpp_wrapper with CMake (build dir: $(CPP_BUILD_DIR))"
	cmake -S . -B $(CPP_BUILD_DIR) -DPELORUS_BUILD_TYPE=$(BUILD_TYPE)
	cmake --build $(CPP_BUILD_DIR) --target cpp_wrapper -- -j

clean:
	rm -f $(EXAMPLES)
	rm -rf $(CPP_BUILD_DIR)
	rm -f cpp_wrapper

run: sync_blocking async_callback
	./sync_blocking
	./async_callback

regression: sync_blocking async_callback cpp_wrapper
	@echo "Running regression: compare async vs sync vs cpp outputs"
	@PELORUS_DUMP_ALL_POINTS=1 python3 compare_async_sync.py

.PHONY: all clean run FORCE

FORCE:
