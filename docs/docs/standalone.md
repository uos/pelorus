# Standalone

Pelorus is build to run on Linux and Mac without additional restrictions. The only dependency is the data source. Given datasets in binary form or your own Mars Rover or SLAM application you can use the C, C++ or Python libraries for an easy integration in existing systems.

To just get Pelorus running on your system for the first time, we will use the SeaKit framework. It was build during development of Pelorus to fix determinism and efficiency from ROS. Let's start using it for getting quick results and then go further with specialized use cases:

Since SeaKit is in active development, we will just build everything from source. That might sound scary but in the Rust world, it is barely an inconvenience. Install [Rust](https://www.rust-lang.org/tools/install) first, then follow the commands.

Install Marina for getting bags:
```sh
cargo install --git git://git.steado.tech/marina --locked

marina pull wachsbleiche/parking_25:hunter
```
We recommend also adding the [tab completions](https://stelzo.codeberg.page/marina/features/shell/completions.html). As with all SeaKit, you also find the docs via `ssh marina@steado.tech`.

Now install Minot, which is used by Pelorus to read bags and process synchronously.
```sh
cargo install --git git://git.steado.tech/minot minot --tag v0.9.1-hiroz --locked
```

(Optional) Most people like fancy TUIs -- and we do as well. But we also offer good old GUIs for visualization. Install [Rerun](https://rerun.io/) via this command:
```sh
cargo install rerun-cli@0.33.1
```

Now we are ready to get our Pelorus odometry with the CLI. 
```sh
cargo install --git ssh://git.steado.tech/pelorus pelorus_cli --features rerun --locked
```

To detect the current shell and install self-refreshing completions for the current user, run:

```sh
pelorus completions
```

Restart the shell afterward. To generate completion files for packaging or a system-wide installation instead, pass the shell explicitly:

=== "Bash"

    === "Linux"

        ~~~bash
        pelorus completions bash | sudo tee /etc/bash_completion.d/pelorus > /dev/null
        ~~~

    === "macOS (Homebrew)"

        ~~~bash
        pelorus completions bash > $(brew --prefix)/etc/bash_completion.d/pelorus
        ~~~

    After installing, reload your shell or source the completion file:

    ~~~bash
    source /etc/bash_completion.d/pelorus
    ~~~

=== "Zsh"

    === "Linux"

        ~~~bash
        mkdir -p ~/.zsh/completions
        pelorus completions zsh > ~/.zsh/completions/_pelorus
        ~~~

    === "macOS (Homebrew)"

        ~~~bash
        pelorus completions zsh > $(brew --prefix)/share/zsh/site-functions/_pelorus
        ~~~

    Add the following lines to your `~/.zshrc` **before** the `compinit` call:

    ~~~bash
    fpath=(~/.zsh/completions $fpath)
    autoload -U compinit && compinit
    ~~~

=== "Fish"

    === "Linux"

        ~~~bash
        pelorus completions fish > ~/.config/fish/completions/pelorus.fish
        ~~~

    === "macOS (Homebrew)"

        ~~~bash
        pelorus completions fish > $(brew --prefix)/share/fish/completions/pelorus.fish
        ~~~

    Fish picks up completions automatically.


That's it. No we will run.
```sh
pelorus offline wachsbleiche/parking_25:hunter -c $(marina resolve wachsbleiche/parking_25:hunter)/pelorus.rl
```

!!! note "GPU Support"

    Pelorus offers various configurations at compile time. You can set them with CMake arguments via uppercasing and setting the prefix `PELORUS_` like so:
    ```sh
    colcon build --packages-select pelorus --cmake-args -D PELORUS_CUDA=ON
    ```


## Standalone and ROS-compatible

You can run Pelorus with ROS 2 Zenoh middleware compatibility as well. Minot and Pelorus both support them.

Install Minot with Hiroz.
```sh
cargo install --git git://git.steado.tech/minot minot --tag v0.9.1-hiroz --no-default-features --features default-hiroz
```

Install Pelorus with Hiroz.
```sh
cargo install --git ssh://git.steado.tech/pelorus pelorus_cli --no-default-features --features rayon,offline-hiroz
```

Run a Zenoh Router.
```sh
cargo install zenohd@1.9.0

zenohd
```

Now run Pelorus in online mode.
```sh
pelorus online -c $(marina resolve wachsbleiche/parking_25:hunter)/pelorus.rl
```

And play a bag with Minot.
```sh
minot async $(marina resolve wachsbleiche/parking_25:hunter) --clock --rate 1.0
```

Now you can open RViz in a ROS-enabled shell and see the results.
```sh
rviz2 -d ./pelorus_cli/rviz/lio.rviz
```
