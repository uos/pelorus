# Pelorus Tasks

Quick commands usually needed for developing Pelorus.

## opt

> Maximum speed

~~~sh
RUSTFLAGS="-C target-cpu=native -C llvm-args=-cost-kind=latency" cargo run --release -- run
~~~


## dev

> Start with development flags and debug info

~~~sh
RUST_BACKTRACE=1 RUST_LOG=debug,lighthouse=off,sea=off,h2=off,hyper_util=off,re_sdk=off,re_chunk=off,tower=off,ratpub=off,tonic=off,tonic_web=off cargo run --release --no-default-features --features dev -- run -c pelorus.rl
~~~
<!-- RUST_LOG=debug,lighthouse=off,sea=off,h2=off,hyper_util=off,re_sdk=off,re_chunk=off,tower=off,ratpub=off cargo run --no-default-features --features dev -- run -->


## setup-rust

> Install rustup

~~~sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
rustup default stable
rustup update
~~~

## load-bag

> Downloads test bag file to current directory

~~~sh
curl -sLo dlg_cut.zip "https://myshare.uni-osnabrueck.de/f/5faf4154af384854ab94?dl=1" && unzip dlg_cut.zip && rm dlg_cut.zip
~~~
