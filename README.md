# Model maker

## Info
* https://coral.ai/docs/accelerator/get-started
* https://coral.ai/docs/edgetpu/compiler/
* https://www.tensorflow.org/lite/models/modify/model_maker/image_classification

## Prerequisites / known good configuration

* macOS Monterey 12.7.3
* EdgeTPU runtime 20221024
* libusb 1.0.27
* Docker Desktop 24.0.2
* Python 3.9.6
* pip 24.0
* pip-tools 7.4.1

## Prepare environment
System stuff
```
brew install libusb
curl -LO https://github.com/google-coral/libedgetpu/releases/download/release-grouper/edgetpu_runtime_20221024.zip
unzip edgetpu_runtime_20221024.zip
cd edgetpu_runtime
sudo bash install.sh
rm -rf edgetpu_runtime*
```
Virtual environment
```
deactivate > /dev/null 2>&1 || :
rm -rf .venv
/usr/bin/python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip==24.0
pip install --upgrade pip-tools==7.4.1
pip-compile
pip install --upgrade -r requirements.txt
```

## Train
```
mkdir -p models
python train.py
```

## Compile to EdgeTPU
Build Docker image
```
cd docker
docker build -t edgetpu:latest .
cd -
```
Start Docker container
```
docker run --rm --interactive --tty --name Edge-TPU-Compiler --mount type=bind,src=./models,dst=/models edgetpu:latest
cd /models
rm -f model_edgetpu.*
edgetpu_compiler model.tflite
exit
```
Delete Docker container (if still present)
```
docker rm $(docker ps -a|grep 'Edge-TPU-Compiler'|awk '{print $1}')
```

## Inference
```
python inference.py
```