# Configuration

Pelorus is configured via [ratslang](https://codeberg.org/stelzo/ratslang). The binary ships the default and minimal configuration, which can be used as a starting point.

You can generate a configuration file template with the CLI or the ROS 2 node:

```sh
# prepend 'ros2 run pelorus' if you only have the ROS 2 package installed 
pelorus config generate recommended > pelorus.rl
```

This will create a minimal configuration file `pelorus.rl` in the current directory that should suffice for most applications. You will need the path to this file to run Pelorus.

For editing your configuration, we recommend using the Ratslang VSCode Extension or the neovim and helix plugins.

After editing, you can validate your configuration against the Pelorus schema using:

```sh
pelorus config validate pelorus.rl
```

If nothing is printed, your file is valid and ready to use.

# TODO maybe generate the default config here from pelorus cli in the ci release

The generated file is the single source of truth and self-documenting. It undergoes type and usage checks at compile time to really make invalid configurations impossible.

If you do not manually validate your edited configuration, Pelorus will do it at startup. If anything goes wrong, it will quit with an error message or give you warnings.
