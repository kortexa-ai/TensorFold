# Pinned ARM64 CUDA 13 toolchain and NCCL. Inference uses only TensorFold's native Zig/CUDA engine.
FROM vllm/vllm-openai@sha256:3f7dd5b777d34d1724456ce71f87385dca288c3bb23029ab27dee358f5d2b971 AS build
ARG ZIG_SHA256=9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8
RUN curl -fsSL https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz -o /tmp/zig.tar.xz \
    && echo "$ZIG_SHA256  /tmp/zig.tar.xz" | sha256sum -c - \
    && mkdir /opt/zig && tar -xf /tmp/zig.tar.xz -C /opt/zig --strip-components=1
WORKDIR /src
COPY build.zig build.zig.zon .zig-version ./
COPY zig ./zig
COPY tools/release ./tools/release
COPY packaging ./packaging
COPY web ./web
RUN /opt/zig/zig build native -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -j4

FROM vllm/vllm-openai@sha256:3f7dd5b777d34d1724456ce71f87385dca288c3bb23029ab27dee358f5d2b971
COPY --from=build /src/zig-out/native/bin/tensorfold-native /usr/local/bin/tensorfold-native
COPY LICENSE NOTICE THIRD_PARTY_NOTICES.md /opt/tensorfold/
COPY LICENSES /opt/tensorfold/LICENSES
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/cuda/lib64:/usr/local/cuda/compat \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENTRYPOINT ["tensorfold-native"]
