docker run -d \
    --name=instella-math-eval \
    --network=host \
    --privileged \
    --device /dev/dri \
    --device /dev/kfd \
    --group-add video \
    --ipc=host \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --shm-size 128G \
    -v $HOME:/dockerx/$HOME \
    -w /dockerx \
    rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4 tail -f /dev/null

