FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

# 基础依赖（含 OpenCV 运行时所需的 libgl1 / libglib2.0-0）
RUN apt-get update && apt-get install -y \
    curl gnupg2 ca-certificates software-properties-common default-jdk git \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda（注意整行不要被截断）
RUN curl -fsSLo /tmp/Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && bash /tmp/Miniconda3.sh -b -p /opt/conda \
 && rm -f /tmp/Miniconda3.sh
ENV PATH="/opt/conda/bin:${PATH}"
RUN /opt/conda/bin/conda clean -a -y || true

# 放入依赖文件（确保这三个文件在 build context 根目录）
RUN mkdir -p /nuplan_devkit
COPY environment.yml /nuplan_devkit/
COPY requirements.txt /nuplan_devkit/
COPY requirements_torch.txt /nuplan_devkit/
ENV NUPLAN_HOME=/nuplan_devkit

# 全局 pip 配置：清华源 + 超时 + 重试 + 优先二进制
RUN printf "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple\ntrusted-host = pypi.tuna.tsinghua.edu.cn\ntimeout = 600\nretries = 10\n" > /etc/pip.conf
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_PROGRESS_BAR=off PIP_PREFER_BINARY=1

# 仅用 conda 部分创建环境：把 environment.yml 中的 `- pip:` 块剔掉，稍后再手动 pip 安装
RUN awk 'BEGIN{p=1} /^ *- pip:/{p=0} p; p==0 && NF==0{p=1}' "$NUPLAN_HOME/environment.yml" > /tmp/env_no_pip.yml \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
 && conda env create -f /tmp/env_no_pip.yml

# 分两步装 pip 依赖；先固定旧版 pip / setuptools，避免 omegaconf==2.1.0rc1 被新 pip 拒绝
RUN conda run -n nuplan python -m pip install "pip<24.1" "setuptools<60" --no-cache-dir --timeout 600 --retries 10 \
 && conda run -n nuplan python -m pip install --no-cache-dir --timeout 600 --retries 10 -r "$NUPLAN_HOME/requirements_torch.txt" \
 && conda run -n nuplan python -m pip install --no-cache-dir --timeout 600 --retries 10 -r "$NUPLAN_HOME/requirements.txt"

# 简单自检
RUN /opt/conda/bin/conda run -n nuplan python -c "import hydra, omegaconf, torch; print('OK', hydra.__version__, omegaconf.__version__, torch.__version__)"

# 默认 PATH 指向 nuplan 环境
ENV PATH="/opt/conda/envs/nuplan/bin:/opt/conda/bin:${PATH}"

WORKDIR /workspace
CMD ["/bin/bash"]
