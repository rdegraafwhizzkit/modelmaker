# Model maker

## Info
https://www.tensorflow.org/lite/models/modify/model_maker/image_classification

## Prerequisites / known good configurations

### Windows 
* Windows 11
* Python 3.9.13
* pip 24.0
* pip-tools 7.4.1

### macOS
* macOS Monterey 12.7.3
* Python 3.9.6
* pip 24.0
* pip-tools 7.4.1 
* brew install libusb

## Prepare environment
```
curl -LO https://github.com/google-coral/libedgetpu/releases/download/release-grouper/edgetpu_runtime_20221024.zip
unzip edgetpu_runtime_20221024.zip
cd edgetpu_runtime
sudo bash install.sh
rm -rf edgetpu_runtime*
```

```
pip install --upgrade pip==24.0
pip install pip-tools
pip-compile
pip install --upgrade -r requirements.txt
```

## Run
```
mkdir -p models
python main.py
```

## Compile
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