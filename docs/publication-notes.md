# Publication notes

The original database plot script contained a plaintext database password. The credential and
connection string are removed from the working tree. Git history is unchanged and still contains
the earlier credential. If it was real, rotate it before making that history public. A history
rewrite requires a separate coordinated decision and is not part of this refactor.

Do not commit `.env`, tokens, downloaded toolchains, caches, or new outputs. The baseline AST
index records symbols/configuration rather than original plotting source and has no database
password. Private university datasets were not copied.

This is an installable research repository without PyPI uploads, deployment services, or an
automatic publication step. GPU checks need NVIDIA hardware; full historical figure reproduction
needs the missing training dataset.
