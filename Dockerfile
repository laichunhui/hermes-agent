FROM ghcr.io/astral-sh/uv:0.11.6-python3.13-trixie@sha256:b3c543b6c4f23a5f2df22866bd7857e5d304b67a564f4feab6ac22044dde719b AS uv_source
FROM tianon/gosu:1.19-trixie@sha256:3b176695959c71e123eb390d427efc665eeb561b1540e82679c15e992006b8b9 AS gosu_source
FROM debian:13.4

# 【核心修复 1】利用原生源先安装 ca-certificates 根证书（不依赖任何国内源，用官方默认网络直接装，非常小）
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

# 【核心修复 2】精确覆写 Debian 13 为清华大学 HTTPS 源（有了证书后，走加密源稳如泰山）
RUN printf "Types: deb\n\
URIs: https://mirrors.tuna.tsinghua.edu.cn/debian/\n\
Suites: trixie trixie-updates\n\
Components: main\n\
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg\n\n\
Types: deb\n\
URIs: https://mirrors.tuna.tsinghua.edu.cn/debian-security/\n\
Suites: trixie-security\n\
Components: main\n\
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg\n" > /etc/apt/sources.list.d/debian.sources

# Disable Python stdout buffering to ensure logs are printed immediately
ENV PYTHONUNBUFFERED=1

# Store Playwright browsers outside the volume mount so the build-time
# install survives the /opt/data volume overlay at runtime.
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/hermes/.playwright

# 2. 分批安装系统依赖，大幅度降低单次构建的内存峰值，防止 OOM (Killed) 
# 第一批：通用基础工具
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ripgrep ffmpeg procps git openssh-client docker-cli tini && \
    rm -rf /var/lib/apt/lists/*

# 第二批：前端运行环境
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    nodejs npm && \
    rm -rf /var/lib/apt/lists/*

# 在这里执行 npm 换源，因为上一轮 npm 已经成功安装好了
RUN npm config set registry https://registry.npmmirror.com/

# 第三批：高耗内存的 C/Python 编译依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential gcc python3 python3-dev libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# Non-root user for runtime; UID can be overridden via HERMES_UID at runtime
RUN useradd -u 10000 -m -d /opt/data hermes

COPY --chmod=0755 --from=gosu_source /gosu /usr/local/bin/
COPY --chmod=0755 --from=uv_source /usr/local/bin/uv /usr/local/bin/uvx /usr/local/bin/

WORKDIR /opt/hermes

# ---------- Layer-cached dependency install ----------
COPY package.json package-lock.json ./
COPY web/package.json web/package-lock.json web/
COPY ui-tui/package.json ui-tui/package-lock.json ui-tui/
COPY ui-tui/packages/hermes-ink/package.json ui-tui/packages/hermes-ink/package-lock.json ui-tui/packages/hermes-ink/

RUN npm install --prefer-offline --no-audit && \
    npx playwright install --with-deps chromium --only-shell && \
    (cd web && npm install --prefer-offline --no-audit) && \
    (cd ui-tui && npm install --prefer-offline --no-audit) && \
    npm cache clean --force

# ---------- Source code ----------
COPY --chown=hermes:hermes . .

# Build browser dashboard and terminal UI assets.
RUN cd web && npm run build && \
    cd ../ui-tui && npm run build && \
    rm -rf node_modules/@hermes/ink && \
    rm -rf packages/hermes-ink/node_modules && \
    cp -R packages/hermes-ink node_modules/@hermes/ink && \
    npm install --omit=dev --prefer-offline --no-audit --prefix node_modules/@hermes/ink && \
    rm -rf node_modules/@hermes/ink/node_modules/react && \
    node --input-type=module -e "await import('@hermes/ink')"

# ---------- Permissions ----------
USER root
RUN chmod -R a+rX /opt/hermes

# ---------- Python virtualenv ----------
RUN uv venv && \
    uv pip install --no-cache-dir -e ".[all]"

# ---------- Runtime ----------
ENV HERMES_WEB_DIST=/opt/hermes/hermes_cli/web_dist
ENV HERMES_HOME=/opt/data
ENV PATH="/opt/data/.local/bin:${PATH}"
VOLUME [ "/opt/data" ]
ENTRYPOINT [ "/usr/bin/tini", "-g", "--", "/opt/hermes/docker/entrypoint.sh" ]