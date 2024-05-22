FROM debian:bookworm-slim
RUN apt-get -y update && \
    apt-get -y install curl gnupg2
RUN curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" > /etc/apt/sources.list.d/coral-edgetpu.list && \
    apt-get -y update && \
    apt-get -y install edgetpu-compiler
CMD bash