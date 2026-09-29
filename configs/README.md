# Configs

The current configuration source of truth is `minitrain/config.py`. Keeping the
small preset registry in Python provides type checking and validation without
adding a YAML dependency or path-sensitive loading logic.

External experiment files may be introduced later only when the number of
benchmark combinations makes them useful.
