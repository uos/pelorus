settings {
  default_registry = glumanda
  registry_timeout = 20s
}

registries {
  glumanda {
    uri = "ssh://marina@131.173.32.180:/home/marina"
    kind = "ssh"
    auth_env = "MARINA_SSH_KEY"
  }
}
