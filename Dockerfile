# Reproducible re-env image for newminhh static binary patcher
# Backed by Arch Linux base-devel
FROM archlinux:base-devel

# Note: Ghidra (/opt/ghidra) and IDA Pro / IDA Free should be bind-mounted at
# runtime (e.g. via -v /opt/ghidra:/opt/ghidra:ro) rather than baked into this
# image for licensing, redistribution, and credential isolation reasons.

RUN pacman -Syu --noconfirm && \
    pacman -S --noconfirm --needed \
        python \
        python-pyelftools \
        python-capstone \
        binutils \
        gcc \
        clang \
        rizin \
        rz-ghidra && \
    pacman -Scc --noconfirm

# Create compatible Python environment expected by newminhh-patch (/opt/re-venv)
RUN python3 -m venv --system-site-packages /opt/re-venv

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONPATH=/opt/newminhh_patcher \
    PATH="/opt/re-venv/bin:${PATH}"

WORKDIR /work

ENTRYPOINT ["/bin/bash"]
