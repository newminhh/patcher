"""newminhh hybrid static binary patcher package (i386 ELF32 -> x32 EM_X86_64 ELF32)."""

__version__ = "1.0.0"


class PatcherError(Exception):
    """Structured fail-loudly exception for ELF validation, RE analysis, or rewriting."""

    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code
